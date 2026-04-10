use std::{
    env, fs,
    io::{BufWriter, Read, Write},
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Duration, SystemTime},
};

use bytes::Bytes;
use color_eyre::eyre::eyre;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures::StreamExt;
use portable_pty::{native_pty_system, ChildKiller, CommandBuilder, MasterPty, PtySize};
use ratatui::{
    backend::Backend,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use tokio::sync::{mpsc, watch};
use tui_term::widget::{Cursor, PseudoTerminal};
use tui_textarea::TextArea;

// ---------------------------------------------------------------------------
// Known agents
// ---------------------------------------------------------------------------

pub struct AgentDef {
    pub name: &'static str,
    pub cmd: &'static str,
    pub args: &'static [&'static str],
    /// How to inject the scratchpad prompt for this agent.
    pub scratchpad_injection: ScratchpadInjection,
}

#[derive(Clone, Copy)]
pub enum ScratchpadInjection {
    /// No injection (shell, unknown agents).
    None,
    /// Set an environment variable with the prompt text.
    EnvVar(&'static str),
    /// Pass a CLI flag with the prompt text (e.g., --system-prompt).
    CliFlag(&'static str),
}

pub const AGENTS: &[AgentDef] = &[
    AgentDef { name: "Claude Code", cmd: "claude", args: &[], scratchpad_injection: ScratchpadInjection::CliFlag("--system-prompt") },
    AgentDef { name: "Hermes", cmd: "hermes", args: &[], scratchpad_injection: ScratchpadInjection::EnvVar("HERMES_EPHEMERAL_SYSTEM_PROMPT") },
    AgentDef { name: "Codex", cmd: "codex", args: &[], scratchpad_injection: ScratchpadInjection::None },
    AgentDef { name: "Pi", cmd: "pi", args: &[], scratchpad_injection: ScratchpadInjection::None },
    AgentDef { name: "Shell", cmd: "", args: &[], scratchpad_injection: ScratchpadInjection::None },
];

fn scratchpad_prompt(path: &PathBuf) -> String {
    format!(
        "You are running inside Lattice, a collaborative workspace. \
         There is a shared scratchpad file at: {}\n\
         This scratchpad is visible to both you and the human in a side panel. \
         You can read and edit it freely. The human can also edit it at the same time. \
         Use it to share notes, plans, code snippets, or anything useful for collaboration.",
        path.display()
    )
}

fn resolve_command(agent: &AgentDef, scratchpad_path: Option<&PathBuf>) -> (String, Vec<String>) {
    if agent.cmd.is_empty() {
        let shell = env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        return (shell, vec![]);
    }

    let mut args: Vec<String> = agent.args.iter().map(|s| s.to_string()).collect();

    // Inject scratchpad via CLI flag if applicable.
    if let (ScratchpadInjection::CliFlag(flag), Some(path)) = (agent.scratchpad_injection, scratchpad_path) {
        args.push(flag.to_string());
        args.push(scratchpad_prompt(path));
    }

    (agent.cmd.to_string(), args)
}

// ---------------------------------------------------------------------------
// Focus
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Agent,
    Terminal,
    Scratchpad,
}

impl Focus {
    fn label(self) -> &'static str {
        match self {
            Focus::Agent => "Agent",
            Focus::Terminal => "Terminal",
            Focus::Scratchpad => "Scratchpad",
        }
    }

    fn next(self) -> Self {
        match self {
            Focus::Agent => Focus::Terminal,
            Focus::Terminal => Focus::Scratchpad,
            Focus::Scratchpad => Focus::Agent,
        }
    }
}

// ---------------------------------------------------------------------------
// Terminal pane (used for both agent and shell)
// ---------------------------------------------------------------------------

struct TerminalPane {
    title: String,
    parser: Arc<RwLock<vt100::Parser>>,
    input_tx: mpsc::Sender<Bytes>,
    master: Box<dyn MasterPty + Send>,
    exit_rx: watch::Receiver<bool>,
    prev_size: (u16, u16),
    exited: bool,
    _killer: Box<dyn ChildKiller + Send + Sync>,
}

impl TerminalPane {
    fn spawn(
        title: &str,
        program: &str,
        args: &[String],
        rows: u16,
        cols: u16,
        extra_env: Vec<(String, String)>,
    ) -> color_eyre::Result<Self> {
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| eyre!(e))?;

        let mut cmd = CommandBuilder::new(program);
        for arg in args {
            cmd.arg(arg);
        }
        cmd.cwd(env::current_dir()?);
        for (key, val) in &extra_env {
            cmd.env(key, val);
        }

        let killer = {
            let mut child = pair.slave.spawn_command(cmd).map_err(|e| eyre!(e))?;
            let killer = child.clone_killer();
            tokio::task::spawn_blocking(move || {
                let _ = child.wait();
            });
            killer
        };
        drop(pair.slave);

        let parser = Arc::new(RwLock::new(vt100::Parser::new(rows, cols, 0)));

        let (exit_tx, exit_rx) = watch::channel(false);
        {
            let parser = parser.clone();
            let mut reader = pair.master.try_clone_reader().map_err(|e| eyre!(e))?;
            tokio::task::spawn_blocking(move || {
                let mut buf = [0u8; 8192];
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            if let Ok(mut p) = parser.write() {
                                p.process(&buf[..n]);
                            }
                        }
                    }
                }
                let _ = exit_tx.send(true);
            });
        }

        let (input_tx, mut input_rx) = mpsc::channel::<Bytes>(64);
        let mut writer = BufWriter::new(pair.master.take_writer().map_err(|e| eyre!(e))?);
        tokio::spawn(async move {
            while let Some(bytes) = input_rx.recv().await {
                if writer.write_all(&bytes).is_err() {
                    break;
                }
                let _ = writer.flush();
            }
        });

        Ok(Self {
            title: title.to_string(),
            parser,
            input_tx,
            master: pair.master,
            exit_rx,
            prev_size: (rows, cols),
            exited: false,
            _killer: killer,
        })
    }

    fn spawn_shell(rows: u16, cols: u16, scratchpad_env: Vec<(String, String)>) -> color_eyre::Result<Self> {
        let shell = env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        Self::spawn("Terminal", &shell, &[], rows, cols, scratchpad_env)
    }

    fn resize(&mut self, rows: u16, cols: u16) {
        if (rows, cols) != self.prev_size && rows > 0 && cols > 0 {
            self.prev_size = (rows, cols);
            if let Ok(mut p) = self.parser.write() {
                p.set_size(rows, cols);
            }
            let _ = self.master.resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            });
        }
    }

    fn render(&self, f: &mut Frame, area: Rect, focused: bool) {
        let border_style = if focused {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let title = if self.exited {
            format!(" {} [exited] ", self.title)
        } else {
            format!(" {} ", self.title)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(border_style);
        let inner = block.inner(area);
        f.render_widget(block, area);

        if let Ok(p) = self.parser.read() {
            let cursor = Cursor::default().visibility(focused && !self.exited);
            let pseudo_term = PseudoTerminal::new(p.screen()).cursor(cursor);
            f.render_widget(pseudo_term, inner);
        }
    }
}

// ---------------------------------------------------------------------------
// Scratchpad pane (file-backed)
// ---------------------------------------------------------------------------

struct ScratchpadPane {
    textarea: TextArea<'static>,
    file_path: PathBuf,
    last_saved: String,
    last_mtime: Option<SystemTime>,
    dirty: bool,
}

impl ScratchpadPane {
    fn new(file_path: PathBuf) -> color_eyre::Result<Self> {
        // Ensure parent dir exists.
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Create the file if it doesn't exist.
        if !file_path.exists() {
            fs::write(&file_path, "")?;
        }

        let content = fs::read_to_string(&file_path)?;
        let mtime = fs::metadata(&file_path).ok().and_then(|m| m.modified().ok());

        let lines: Vec<String> = if content.is_empty() {
            vec![String::new()]
        } else {
            content.lines().map(|l| l.to_string()).collect()
        };

        let mut textarea = TextArea::new(lines);
        textarea.set_cursor_line_style(Style::default());

        Ok(Self {
            textarea,
            file_path,
            last_saved: content,
            last_mtime: mtime,
            dirty: false,
        })
    }

    fn path_display(&self) -> String {
        self.file_path.display().to_string()
    }

    fn content(&self) -> String {
        self.textarea.lines().join("\n")
    }

    fn save(&mut self) {
        let content = self.content();
        if content != self.last_saved {
            if fs::write(&self.file_path, &content).is_ok() {
                self.last_saved = content;
                self.last_mtime =
                    fs::metadata(&self.file_path).ok().and_then(|m| m.modified().ok());
                self.dirty = false;
            }
        }
    }

    /// Check if the file was modified externally and reload if so.
    /// Only reloads when the scratchpad is NOT focused (to avoid fighting user edits).
    fn check_external_changes(&mut self, focused: bool) {
        if focused {
            return;
        }
        let current_mtime = fs::metadata(&self.file_path).ok().and_then(|m| m.modified().ok());
        if current_mtime != self.last_mtime {
            if let Ok(content) = fs::read_to_string(&self.file_path) {
                if content != self.last_saved {
                    let lines: Vec<String> = if content.is_empty() {
                        vec![String::new()]
                    } else {
                        content.lines().map(|l| l.to_string()).collect()
                    };
                    self.textarea = TextArea::new(lines);
                    self.textarea.set_cursor_line_style(Style::default());
                    self.last_saved = content;
                    self.dirty = false;
                }
                self.last_mtime = current_mtime;
            }
        }
    }

    fn handle_input(&mut self, event: &Event) {
        self.textarea.input(event.clone());
        self.dirty = true;
    }

    fn render(&mut self, f: &mut Frame, area: Rect, focused: bool) {
        let border_style = if focused {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let lines = self.textarea.lines().len();
        let dirty_marker = if self.dirty { " *" } else { "" };
        let title = format!(" Scratchpad ({} lines){} — {} ", lines, dirty_marker, self.path_display());
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(border_style);
        self.textarea.set_block(block);

        if focused {
            self.textarea
                .set_cursor_style(Style::default().bg(Color::White).fg(Color::Black));
        } else {
            self.textarea.set_cursor_style(Style::default());
        }

        f.render_widget(&self.textarea, area);
    }

    fn cleanup(&self) {
        let _ = fs::remove_file(&self.file_path);
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

pub struct App {
    focus: Focus,
    agent_pane: TerminalPane,
    shell_pane: TerminalPane,
    scratchpad: ScratchpadPane,
    #[allow(dead_code)]
    session_id: String,
}

fn generate_session_id() -> String {
    use std::time::UNIX_EPOCH;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let pid = std::process::id();
    format!("{ts}-{pid}")
}

fn scratchpad_path(session_id: &str) -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".lattice")
        .join("sessions")
        .join(format!("{session_id}-scratchpad.md"))
}

impl App {
    pub fn new<B: Backend>(terminal: &Terminal<B>, agent_index: usize) -> color_eyre::Result<Self> {
        let size = terminal.size()?;

        let agent = &AGENTS[agent_index];

        let session_id = generate_session_id();
        let scratch_path = scratchpad_path(&session_id);
        let scratchpad = ScratchpadPane::new(scratch_path)?;

        // Build agent command with scratchpad injection.
        let (program, args) = resolve_command(agent, Some(&scratchpad.file_path));

        // Build env vars for the agent process.
        let mut agent_env = vec![
            ("LATTICE_SCRATCHPAD".to_string(), scratchpad.file_path.display().to_string()),
        ];
        if let ScratchpadInjection::EnvVar(var) = agent.scratchpad_injection {
            agent_env.push((var.to_string(), scratchpad_prompt(&scratchpad.file_path)));
        }

        // Shell just gets the path env var.
        let shell_env = vec![
            ("LATTICE_SCRATCHPAD".to_string(), scratchpad.file_path.display().to_string()),
        ];

        // Agent pane: left 50%.
        let agent_cols = (size.width * 50 / 100).saturating_sub(2);
        let agent_rows = size.height.saturating_sub(3);

        // Shell pane: bottom-right.
        let shell_cols = (size.width * 50 / 100).saturating_sub(2);
        let shell_rows = (size.height * 60 / 100).saturating_sub(3);

        Ok(Self {
            focus: Focus::Agent,
            agent_pane: TerminalPane::spawn(agent.name, &program, &args, agent_rows, agent_cols, agent_env)?,
            shell_pane: TerminalPane::spawn_shell(shell_rows, shell_cols, shell_env)?,
            scratchpad,
            session_id,
        })
    }

    #[allow(dead_code)]
    pub fn scratchpad_path(&self) -> &PathBuf {
        &self.scratchpad.file_path
    }

    pub async fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> color_eyre::Result<()> {
        let mut events = EventStream::new();
        let mut tick_interval = tokio::time::interval(Duration::from_millis(50));
        let mut save_interval = tokio::time::interval(Duration::from_millis(500));

        loop {
            terminal.draw(|f| self.render(f))?;

            tokio::select! {
                maybe_event = events.next() => {
                    let Some(Ok(event)) = maybe_event else { break };
                    if self.handle_event(event).await {
                        break;
                    }
                }
                _ = self.agent_pane.exit_rx.changed() => {
                    self.agent_pane.exited = true;
                }
                _ = self.shell_pane.exit_rx.changed() => {
                    self.shell_pane.exited = true;
                }
                _ = save_interval.tick() => {
                    // Auto-save dirty scratchpad and check for external changes.
                    if self.scratchpad.dirty {
                        self.scratchpad.save();
                    }
                    self.scratchpad.check_external_changes(self.focus == Focus::Scratchpad);
                }
                _ = tick_interval.tick() => {}
            }
        }

        // Final save and cleanup.
        self.scratchpad.save();
        self.scratchpad.cleanup();

        Ok(())
    }

    async fn handle_event(&mut self, event: Event) -> bool {
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                // --- Global intercepts ---
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.code == KeyCode::Char('q')
                {
                    return true;
                }
                if key.modifiers.contains(KeyModifiers::ALT) {
                    match key.code {
                        KeyCode::Char('1') => {
                            self.focus = Focus::Agent;
                            return false;
                        }
                        KeyCode::Char('2') => {
                            self.focus = Focus::Terminal;
                            return false;
                        }
                        KeyCode::Char('3') => {
                            self.focus = Focus::Scratchpad;
                            return false;
                        }
                        _ => {}
                    }
                }
                if key.code == KeyCode::BackTab {
                    self.focus = self.focus.next();
                    return false;
                }

                // --- Dispatch to focused pane ---
                match self.focus {
                    Focus::Agent => {
                        if !self.agent_pane.exited {
                            if let Some(bytes) = key_event_to_bytes(&key) {
                                let _ = self.agent_pane.input_tx.send(Bytes::from(bytes)).await;
                            }
                        }
                    }
                    Focus::Terminal => {
                        if !self.shell_pane.exited {
                            if let Some(bytes) = key_event_to_bytes(&key) {
                                let _ = self.shell_pane.input_tx.send(Bytes::from(bytes)).await;
                            }
                        }
                    }
                    Focus::Scratchpad => {
                        self.scratchpad.handle_input(&Event::Key(key));
                    }
                }
            }
            Event::Resize(_cols, _rows) => {}
            _ => {}
        }
        false
    }

    fn render(&mut self, f: &mut Frame) {
        let area = f.area();

        let [main_area, status_area] =
            Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(area);

        // Three-pane: agent (left 50%) | right (50%).
        let [agent_area, right_area] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(main_area);

        // Right: scratchpad (top 40%) | terminal (bottom 60%).
        let [scratch_area, term_area] =
            Layout::vertical([Constraint::Percentage(70), Constraint::Percentage(30)])
                .areas(right_area);

        // Resize PTYs if inner dimensions changed.
        let agent_inner = Block::default().borders(Borders::ALL).inner(agent_area);
        self.agent_pane.resize(agent_inner.height, agent_inner.width);

        let term_inner = Block::default().borders(Borders::ALL).inner(term_area);
        self.shell_pane.resize(term_inner.height, term_inner.width);

        // Render panes.
        self.agent_pane
            .render(f, agent_area, self.focus == Focus::Agent);
        self.scratchpad
            .render(f, scratch_area, self.focus == Focus::Scratchpad);
        self.shell_pane
            .render(f, term_area, self.focus == Focus::Terminal);

        // Status bar.
        let status_text = format!(
            " Alt+1: Agent | Alt+2: Terminal | Alt+3: Scratchpad | Shift+Tab: Cycle | Ctrl+Q: Quit  [{}]",
            self.focus.label()
        );
        let status = Paragraph::new(status_text)
            .style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(Alignment::Left);
        f.render_widget(status, status_area);
    }
}

// ---------------------------------------------------------------------------
// Key translation
// ---------------------------------------------------------------------------

fn key_event_to_bytes(key: &KeyEvent) -> Option<Vec<u8>> {
    let bytes = match key.code {
        KeyCode::Char(ch) => {
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                let upper = ch.to_ascii_uppercase();
                match upper {
                    '2' | '@' | ' ' => vec![0],
                    '3' | '[' => vec![27],
                    '4' | '\\' => vec![28],
                    '5' | ']' => vec![29],
                    '6' | '^' => vec![30],
                    '7' | '-' | '_' => vec![31],
                    c if ('A'..='_').contains(&c) => vec![c as u8 - 64],
                    _ => vec![ch as u8],
                }
            } else {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                s.as_bytes().to_vec()
            }
        }
        KeyCode::Enter => vec![b'\r'],
        KeyCode::Backspace => vec![8],
        KeyCode::Tab => vec![9],
        KeyCode::Esc => vec![27],
        KeyCode::Left => vec![27, 91, 68],
        KeyCode::Right => vec![27, 91, 67],
        KeyCode::Up => vec![27, 91, 65],
        KeyCode::Down => vec![27, 91, 66],
        KeyCode::Home => vec![27, 91, 72],
        KeyCode::End => vec![27, 91, 70],
        KeyCode::PageUp => vec![27, 91, 53, 126],
        KeyCode::PageDown => vec![27, 91, 54, 126],
        KeyCode::BackTab => vec![27, 91, 90],
        KeyCode::Delete => vec![27, 91, 51, 126],
        KeyCode::Insert => vec![27, 91, 50, 126],
        KeyCode::F(n) => match n {
            1 => vec![27, 79, 80],
            2 => vec![27, 79, 81],
            3 => vec![27, 79, 82],
            4 => vec![27, 79, 83],
            5 => vec![27, 91, 49, 53, 126],
            6 => vec![27, 91, 49, 55, 126],
            7 => vec![27, 91, 49, 56, 126],
            8 => vec![27, 91, 49, 57, 126],
            9 => vec![27, 91, 50, 48, 126],
            10 => vec![27, 91, 50, 49, 126],
            11 => vec![27, 91, 50, 51, 126],
            12 => vec![27, 91, 50, 52, 126],
            _ => return None,
        },
        _ => return None,
    };
    Some(bytes)
}
