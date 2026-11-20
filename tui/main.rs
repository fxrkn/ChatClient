use std::io::stdout;

use crossterm::{
	event::{read, Event, KeyCode},
	terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
	cursor::MoveTo,
	ExecutableCommand,
};

use ratatui::{
	backend::CrosstermBackend, 
	widgets::{Clear as ClearWidget, Block, Borders, Paragraph, List, ListItem}, 
	Terminal,
};

use ratatui::style::{
	Color,
	Style
};

use ratatui::layout::{
	Layout, 
	Constraint, 
	Direction
};

struct App{
	sekme: usize,
}

fn main() -> std::io::Result<()>{

	let mut out = stdout();
	out.execute(Clear(ClearType::All))?.execute(MoveTo(0, 0))?;
	enable_raw_mode()?;

	let mut app = App{
		sekme: 0,
	};

	let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

	loop{

		terminal.draw(|frame|{
			let ana_parcalar = Layout::default()
			.direction(Direction::Vertical)
			.constraints([Constraint::Min(1), Constraint::Length(1),])
			.split(frame.size());

			let cerceve = Block::default().borders(Borders::ALL);

			let sekmeler = ["asd1", "asd2", "asd3"];
			let ogeler: Vec<ListItem> = sekmeler.iter().enumerate().map(|(i, &isim)| {
				if i == app.sekme {
					ListItem::new(format!("> {}", isim)).style(Style::default().fg(Color::Black).bg(Color::White))
				} else {
					ListItem::new(format!("  {}", isim))
				}
			}).collect();	

			let liste = List::new(ogeler).block(cerceve);
			frame.render_widget(ClearWidget, frame.size());
			frame.render_widget(liste, ana_parcalar[0]);

			let alt_bilgi = Paragraph::new(" [F1] quit");
			frame.render_widget(alt_bilgi, ana_parcalar[1]);
		})?;

		if let Event::Key(key_binds) = read()?{
			if key_binds.code == KeyCode::F(1){
				break;
			}

			if key_binds.code == KeyCode::Tab{
				app.sekme = (app.sekme + 1) % 3;
			}

			if key_binds.code == KeyCode::Down {
				app.sekme = (app.sekme + 1) % 3;
			}

			if key_binds.code == KeyCode::Up {
				app.sekme = (app.sekme + 2) % 3;
			}
		}
	}

	disable_raw_mode()?;
	let mut out = stdout();
	out.execute(Clear(ClearType::All))?.execute(MoveTo(0, 0))?;
	Ok(())
}






