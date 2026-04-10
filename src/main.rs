mod app;

use std::io::{self, BufWriter};

use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use app::{App, AGENTS};

/// Drop guard that restores the terminal on any exit path (normal, error, panic).
struct TerminalGuard;

impl TerminalGuard {
    fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

fn select_agent() -> usize {
    let args: Vec<String> = std::env::args().collect();

    // Check for --agent <name> flag.
    if let Some(pos) = args.iter().position(|a| a == "--agent") {
        if let Some(name) = args.get(pos + 1) {
            let name_lower = name.to_lowercase();
            if let Some(idx) = AGENTS.iter().position(|a| a.name.to_lowercase() == name_lower || a.cmd == name_lower) {
                return idx;
            }
            eprintln!("Unknown agent: {}", name);
            eprintln!("Available agents:");
            for a in AGENTS {
                eprintln!("  {} ({})", a.name, if a.cmd.is_empty() { "$SHELL" } else { a.cmd });
            }
            std::process::exit(1);
        }
    }

    // No flag: print menu to stderr and read choice.
    eprintln!("Select an agent:");
    for (i, agent) in AGENTS.iter().enumerate() {
        let cmd = if agent.cmd.is_empty() { "$SHELL" } else { agent.cmd };
        eprintln!("  {}) {} ({})", i + 1, agent.name, cmd);
    }
    eprint!("Choice [1-{}]: ", AGENTS.len());

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    let choice: usize = input.trim().parse().unwrap_or(0);
    if choice == 0 || choice > AGENTS.len() {
        eprintln!("Invalid choice, defaulting to Shell");
        AGENTS.len() - 1 // Shell is last
    } else {
        choice - 1
    }
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let agent_index = select_agent();
    eprintln!("Launching {}...", AGENTS[agent_index].name);

    let _guard = TerminalGuard::new()?;

    let backend = CrosstermBackend::new(BufWriter::new(io::stdout()));
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(&terminal, agent_index)?;
    app.run(&mut terminal).await?;

    Ok(())
}
