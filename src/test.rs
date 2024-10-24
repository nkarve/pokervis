use pokervis::card::*;
use pokervis::holdem::*;

fn main() {
    let mut game = HoldemGame::new(vec![
        Player::new("ALC".to_string()),
        Player::new("BOB".to_string()),
    ]);

    game.play_rounds(50, true);
}
