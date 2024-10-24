use crate::card::*;
use std::cell::{Cell, RefCell};
use std::io::{self, Write};
use std::os::windows::thread;
use std::rc::Rc;
use itertools::Itertools;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand::{seq::SliceRandom, seq::index::sample_weighted, distributions::WeightedIndex, thread_rng};

use std::net::{SocketAddr, UdpSocket};
use std::time::{SystemTime, Instant, Duration};
use renet::transport::{NetcodeServerTransport, ServerConfig, ServerAuthentication};
use renet::{RenetServer, ConnectionConfig, ServerEvent, DefaultChannel};

// this will handle server etc. so decoupled from game logic
pub struct HoldemGame {
    players: Vec<Player>,
    // roundhistory
}

impl HoldemGame {
    pub fn new(players: Vec<Player>) -> Self {
        HoldemGame {
            players
        }
    }

    pub fn play_rounds(&mut self, n: u32, wait_for_user: bool) {       
        /* let mut server = RenetServer::new(ConnectionConfig::default());
        let server_addr: SocketAddr = "127.0.0.1:6030".parse().unwrap();

        let socket: UdpSocket = UdpSocket::bind(server_addr).unwrap();
        let server_config = ServerConfig {
            current_time: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap(),
            max_clients: 64,
            protocol_id: 0,
            public_addresses: vec![server_addr],
            authentication: ServerAuthentication::Unsecure
        };
        let mut transport = NetcodeServerTransport::new(server_config, socket).unwrap();
        
        println!("Server started on {}", server_addr);

        let mut last_updated = Instant::now();
        loop {
            let now = Instant::now();
            server.update(now - last_updated);
            transport.update(now - last_updated, &mut server).unwrap();
            last_updated = now;

            while let Some(event) = server.get_event() {
                match event {
                    ServerEvent::ClientConnected { client_id } => {
                        println!("Client {client_id} connected");
                    }
                    ServerEvent::ClientDisconnected { client_id, reason } => {
                        println!("Client {client_id} disconnected: {reason}");
                    }
                }
            }

            for client_id in server.clients_id() {
                while let Some(message) = server.receive_message(client_id, DefaultChannel::ReliableOrdered) {
                    println!("Received message from client {}: {:?}", client_id, message);
                    server.broadcast_message(DefaultChannel::ReliableOrdered, message);
                }
            }

            transport.send_packets(&mut server);

            std::thread::sleep(Duration::from_millis(50));
        } */


        for _ in 0..n {
            let mut round = HoldemRound::new(self.players.clone(), 1, 2);
            round.play();
            self.players = round.players;

            if wait_for_user {
                println!("Press Enter to continue to next round...");
                io::stdin().read_line(&mut String::new()).unwrap();
            }

            self.players.rotate_left(1);
        }
    }
}

pub struct RoundData {
    pot: u32,
    board: Vec<Card>,
    small_blind: u32,
    big_blind: u32,
    current_bet: u32,
    current_idx: usize,
    button: usize,
    active_count: usize,
    actions: Vec<(String, BetAction)>,
    // last_action: BetAction,
}

struct HoldemRound {
    players: Vec<Player>,
    deck: Vec<Card>,
    data: RoundData,
    // betting_actions: Vec<BetAction>,
}

impl HoldemRound {
    fn new(mut players: Vec<Player>, small_blind: u32, big_blind: u32) -> Self {
        let mut rng = thread_rng();
        let mut deck = DECK.to_vec();
        deck.shuffle(&mut rng);
        
        for player in players.iter_mut() {
            let card1 = deck.pop().unwrap();
            let card2 = deck.pop().unwrap();
            player.hole_cards = vec![card1, card2];

            // TODO: remove
            player.ai = Some(AI::new(player.name.clone(), player.hole_cards.clone()));
        }

        let n = players.len();

        HoldemRound {
            players,
            deck,
            data: RoundData {
                pot: 0,
                board: Vec::new(),
                small_blind,
                big_blind,
                current_bet: 0,
                current_idx: 1,
                button: 0,
                active_count: n,
                actions: Vec::new(),
            }
            // betting_actions: Vec::new(),
        }
    }

    fn betting_round(&mut self) {        
        self.data.current_idx = self.data.button;
        let starting_pot = self.data.pot; 

        loop {
            if self.data.active_count == 1 {
                break;
            }

            self.data.current_idx += 1;
            let idx = self.data.current_idx % self.players.len();
            let player = &mut self.players[idx];

            match player.last_action {
                Some(BetAction::Bet(amount)) => if amount == self.data.current_bet {
                    break;
                },
                Some(BetAction::Fold) => continue,
                None => {},
                _ => if self.data.pot == starting_pot {
                    break;
                },
            }

            if player.stack <= 0 { continue; }
            
            player.ai.as_mut().unwrap().update_ranges(&self.data);
            // TODO: assume legal for now but add legality checking
            let action = player.get_bet_action(&self.data);
            
            println!(">> Player {} {}", player.name, match action {
                BetAction::Fold => String::from("folds"),
                BetAction::Check => String::from("checks"),
                BetAction::Bet(x) => {
                    if x > self.data.current_bet {
                        format!("raises to {}", x)
                    } else {
                        format!("calls {}", x)
                    }
                }
            });

            self.apply_action(idx, action);
        }

        self.data.current_idx = 1;
        self.data.current_bet = 0;

        for player in &mut self.players {
            if player.last_action != Some(BetAction::Fold) {
                player.last_action = None;
            }
        }
    }

    fn showdown(&mut self) {
        let shows = self.players.iter()
            .map(|p| p.get_showdown_action(&self.data))
            .collect_vec();

        let binding = self.players.clone();
        let best_hands = binding.iter().enumerate()
            .filter(|(i, p)| p.last_action != Some(BetAction::Fold) && shows[*i])
            .map(|(i, p)| ((i, p), eval7([self.data.board.as_slice(), &p.hole_cards].concat())))
            .collect_vec();
            
        let winning_score = best_hands.iter().map(|(_, (score, _))| score).min().unwrap();
        let winners_idx = best_hands.iter()
            .filter(|(_, (score, _))| score == winning_score)
            .map(|((i, _), _)| i)
            .collect_vec();
        

        if self.data.active_count > 1 {
            println!("\n\n\x1b[1mSHOWDOWN!\x1b[0m\n");
        }
        for (i, p) in self.players.iter().enumerate() {
            if p.last_action != Some(BetAction::Fold) { print!("\x1b[94m"); }
            print!("\x1b[1m{}:\x1b[0m ", p.name); 
            
            if shows[i] /* && p.last_action != Some(BetAction::Fold) */ {
                for card in &p.hole_cards { print!("{card} "); }
            } else {
                print!("XX XX ");
            };
            println!();
        }
        println!();

        for i in &winners_idx {
            let amt = self.data.pot as i32 / winners_idx.len() as i32;
            self.players[**i].stack += amt;
            println!("\x1b[92m\x1b[1mPlayer {} wins ${}\x1b[0m", self.players[**i].name, amt);
        }
        println!("\n");

        // TODO: rewrite this as imperative
        // TODO: side pot
        /* let mut winning_score = 9999;
        for p in &self.players {
            if p.last_action == Some(BetAction::Fold) || !shows.next().unwrap() { continue; }
            let (score, _) = eval7([self.data.board.as_slice(), &p.hole_cards].concat());
            winning_score = winning_score.min(score);
        }

        let mut winning_idx = Vec::new();
        for (i, p) in self.players.iter().enumerate() {
            if p.last_action == Some(BetAction::Fold) || !shows.next().unwrap() { continue; }
            let (score, _) = eval7([self.data.board.as_slice(), &p.hole_cards].concat());
            if score == winning_score {
                winning_idx.push(i);
            }
        }

        for i in &winning_idx {
            let amt = self.data.pot as i32 / winning_idx.len() as i32;
            self.players[*i].stack += amt;
            println!("Player {} wins ${}", self.players[*i].name, amt);
        } */
    }
    
    fn apply_action(&mut self, player_idx: usize, action: BetAction) {
        let player = &mut self.players[player_idx];

        match action {
            BetAction::Bet(amount) => {
                self.data.pot += amount;
                player.stack -= amount as i32;
                self.data.current_bet = self.data.current_bet.max(amount);

                if let Some(BetAction::Bet(x)) = player.last_action {
                    self.data.pot -= x;
                    player.stack += x as i32;
                }
            },
            BetAction::Fold => self.data.active_count -= 1,
            _ => {}
        }

        player.last_action = Some(action);
        self.data.actions.push((player.name.clone(), action));
    }

    fn play(&mut self) {
        self.apply_action(self.data.current_idx, BetAction::Bet(self.data.small_blind));
        self.data.current_idx = (self.data.current_idx + 1) % self.players.len();
        self.apply_action(self.data.current_idx, BetAction::Bet(self.data.big_blind));

        self.print();
        self.betting_round();

        let draws = [3, 1, 1];
        for draw in draws.iter() {
            for _ in 0..*draw {
                self.data.board.push(self.deck.pop().unwrap());
            }

            self.print();
            self.betting_round();
        }

        self.showdown();
        self.reset();
    }

    // TODO: augment this if merging HoldemGame and HoldemRound
    fn reset(&mut self) {
        for player in &mut self.players {
            player.hole_cards.clear();
            player.last_action = None;
        }
    }

    fn print(&self) {
        print!("\n\n|| ");
        for card in &self.data.board {
            print!("{card} ");
        }
        for _ in 0..(5 - self.data.board.len()) {
            print!("XX ");
        }
        println!("||\n\n --- Pot: ${} ---", self.data.pot);
        println!("     Bet: ${}\n", self.data.current_bet);

        for (i, p) in self.players.iter().enumerate() {
            if p.last_action != Some(BetAction::Fold) { print!("\x1b[94m"); }
            print!("\x1b[1m{}:\x1b[0m ", p.name); 
            
            /* for card in &p.hole_cards { print!("{card} "); }
            println!("\x1b[93m-- ${}\x1b[0m", p.stack);
            */
            
            if p.name == "ALC" {
                for card in &p.hole_cards { print!("{card} "); }
                println!("\x1b[93m-- ${}\x1b[0m", p.stack);
            } else {
                print!("XX XX ");
                println!("-- ${}", p.stack);
            };
        }
        println!("\n");
    }

}

trait PlayerTrait {
    fn get_bet_action(&self, data: &RoundData) -> BetAction;
    fn get_showdown_action(&self, data: &RoundData) -> bool;
}

// TODO: allow 3+ players
#[derive(Debug, Clone)]
pub struct AI {
    name: String,
    stack: i32,
    hole_cards: Vec<Card>,
    last_action: Option<BetAction>,
    rng: RefCell<ThreadRng>, // TODO: change to fast rng
    his_range: Vec<f32>,
    his_wins: RefCell<Vec<f32>>, // TODO: store wins only for previous round
    comm_range: Vec<f32>,
    mask: Vec<bool>,
}

impl AI {
    pub fn new(name: String, hole_cards: Vec<Card>) -> Self {
        let mut mask = vec![true; 52];
        for card in &hole_cards {
            mask[card.idx()] = false;
        }
        
        AI {
            name,
            stack: 500,
            hole_cards,
            last_action: None,
            rng: RefCell::new(thread_rng()),
            his_range: vec![3.; 169],
            his_wins: RefCell::new(vec![0.; 169]),
            comm_range: vec![3.; 52],
            mask
        }
    }

    pub fn set_hole_cards(&mut self, hole_cards: Vec<Card>) {
        self.hole_cards = hole_cards;
        self.mask = vec![true; 52];

        for card in &self.hole_cards {
            self.mask[card.idx()] = false;
        }
    }

    fn draw_n(&self, range: &Vec<f32>, mask: &mut Vec<bool>, n: usize) -> Vec<Card> {
        let idx = sample_weighted(&mut *self.rng.borrow_mut(), 52, |x| if mask[x] {range[x]} else {0.}, n)
            .unwrap()
            .into_iter();

        idx.map(|x| {mask[x] = false; DECK[x]}).collect()
    }

    fn draw2(&self, range: &Vec<f32>, mask: &mut Vec<bool>) -> (usize, Vec<Card>) {
        // TODO: this method means that if e.g. the board has AA, all Ax combos
        // are suddenly rarer even though real win rate is higher
        // This could still be fine -- of course you don't want to lerp via win rate
        // conditional on selection because this nullifies the effect of preflop ranges
        
        let (s1, s2) = SUIT_PAIRS.choose(&mut *self.rng.borrow_mut()).unwrap();
        
        let idx = sample_weighted(&mut *self.rng.borrow_mut(), 169, |x| if mask[Card::new(RANK_PAIRS[x].0, *s1).idx()] && mask[Card::new(RANK_PAIRS[x].1, *s2).idx()] {range[x]} else {0.}, 1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        
        let card1 = Card::new(RANK_PAIRS[idx].0, *s1);
        let card2 = Card::new(RANK_PAIRS[idx].1, *s2);

        mask[card1.idx()] = false;
        mask[card2.idx()] = false;

        (idx, vec![card1, card2])
    }

    pub fn equity(&self, samples: u32, board: &[Card]) -> f32 {
        let mut wins = 0.;
        let mut his_wins = vec![0.; 169];
        let mut selections = vec![0; 169];

        for _ in 0..samples {
            let mut mask = self.mask.clone();
            
            let (idx, his_hand) = self.draw2(&self.his_range, &mut mask);
            let comm = self.draw_n(&self.comm_range, &mut mask, 5 - board.len()); 

            let my_score = eval7([board, comm.as_slice(), self.hole_cards.as_ref()].concat()).0;
            let his_score = eval7([board, comm.as_slice(), his_hand.as_ref()].concat()).0;

            if my_score < his_score {
                wins += 1.;
            } else if my_score == his_score {
                wins += 0.5;
                his_wins[idx] += 0.5;
            } else {
                his_wins[idx] += 1.;
            }
            selections[idx] += 1;
        }

        his_wins = his_wins.iter().zip(selections.iter()).map(|(x, y)| x / *y as f32).collect();
        print_pair_range(|i| (6. * his_wins[i]) as u8, 34);
        self.his_wins.replace(his_wins);
        
        return wins / samples as f32;
    }

    // pot here is the old pot size before having added bet
    fn pot_odds(pot: u32, bet: u32) -> f32 {
        bet as f32 / (pot + bet) as f32
    }

    fn size_bet(&self, pot_odds: f32, equity: f32, data: &RoundData) -> u32 {
        // TODO: implement value bets, bluffs
        let val_equity = if equity > 0.75 { equity / 1.5 } else { equity };

        let max_bet = (data.pot as f32 / (1. / val_equity - 1.)) as u32;
        let min_raise = data.current_bet * 2;
        let my_bet = (data.current_bet.max(data.big_blind) + max_bet) / 2;

        if my_bet < min_raise {
            data.current_bet
        } else {
            my_bet
        }
    }

    fn update_ranges(&mut self, data: &RoundData) {
        if data.board.len() > 1 && self.mask[data.board[0].idx()] {
            for i in 0..169 {
                self.his_range[i] = if PREFLOP_EQUITIES[i] > 0.5 {4.} else {1.5};
            }
        }
        
        if data.actions.last().is_some_and(|x| x.0 != self.name &&
            match x.1 {
                BetAction::Bet(x) => x > data.big_blind * 2,
                _ => false 
            }) {
                let alpha = 2.;
                for i in 0..169 {
                    // TODO: change (- 0.5) to (- top x% equity depending on bet size)
                    self.his_range[i] = self.his_range[i] as f32 + alpha * (self.his_wins.borrow()[i] - 0.5);
                    self.his_range[i] = self.his_range[i].clamp(0.1, 6.);
                }
        }

        for card in &data.board {
            self.mask[card.idx()] = false;
        }
    }
}

impl PlayerTrait for AI {
    fn get_bet_action(&self, data: &RoundData) -> BetAction {
        print_pair_range(|i| self.his_range[i] as u8, 39);
        
        // actually these are call pot odds only
        let pot_odds = AI::pot_odds(data.pot, data.current_bet.max(data.big_blind));

        let equity = self.equity(50000, data.board.as_slice());
        // println!("{} equity: {}", self.name, equity);

        if equity > pot_odds {
            return BetAction::Bet(self.size_bet(pot_odds, equity, data));
        } else {
            return BetAction::Fold;
        }
    }

    fn get_showdown_action(&self, data: &RoundData) -> bool {
        true //data.active_count != 1
    }
}

// server side internal representation of player. communicates with client side via TCP socket
// TODO: refactor this by splitting into PlayerID features and per-player round data features
#[derive(Debug, Clone)]
pub struct Player {
    name: String,
    //socket
    stack: i32,
    hole_cards: Vec<Card>,
    last_action: Option<BetAction>,
    ai: Option<AI>,
}

impl Player {
    pub fn new(name: String) -> Self {
        Player {
            name,
            stack: 500,
            hole_cards: Vec::new(),
            last_action: None,
            ai: None,
        }
    }

    fn get_random_action(&self, data: &RoundData) -> BetAction {
        let mut rng = thread_rng();
        let mut actions = vec![BetAction::Fold, BetAction::Bet(data.current_bet), BetAction::Bet(data.current_bet + 2)];

        if data.current_bet == 0 {
            actions.push(BetAction::Check);
        }

        *actions.choose(&mut rng).unwrap()
    }

    fn get_action_human(&self, data: &RoundData) -> BetAction {
        loop {
            print!("Player {}'s Action: ", self.name);
            io::stdout().flush().unwrap();

            let mut line = String::new();
            let _ = io::stdin().read_line(&mut line);

            line = line.to_ascii_lowercase();

            return match line.chars().next().unwrap() {
                'f' => BetAction::Fold,
                'k' => BetAction::Check,
                'b' => BetAction::Bet(line[1..].trim_end().parse().unwrap()),
                '1'..='9' => BetAction::Bet(line.trim_end().parse().unwrap()),
                'l' => BetAction::Bet(data.current_bet),
                'd' => BetAction::Bet(data.current_bet * 2),
                'q' => BetAction::Bet(data.current_bet * 4),
                _ => continue
            };
        }
    }
}
impl PlayerTrait for Player {
    //TODO: change this obviously
    fn get_bet_action(&self, data: &RoundData) -> BetAction {
        if self.name == "ALC" {
            return self.get_action_human(data);
        } else {
            let action = self.ai.as_ref().unwrap().get_bet_action(data);
            return action;
        }
    }

    fn get_showdown_action(&self, data: &RoundData) -> bool {
        true
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BetAction {
    Fold,
    Check,
    Bet(u32),
}


pub const SUIT_PAIRS_2: [(Suit, Suit); 10] = [
    (Suit::Clubs, Suit::Clubs), (Suit::Clubs, Suit::Diamonds), (Suit::Clubs, Suit::Hearts), (Suit::Clubs, Suit::Spades),
    (Suit::Diamonds, Suit::Diamonds), (Suit::Diamonds, Suit::Hearts), (Suit::Diamonds, Suit::Spades),
    (Suit::Hearts, Suit::Hearts), (Suit::Hearts, Suit::Spades),
    (Suit::Spades, Suit::Spades)
];

pub const SUIT_PAIRS: [(Suit, Suit); 16] = [
    (Suit::Clubs, Suit::Clubs), (Suit::Clubs, Suit::Diamonds), (Suit::Clubs, Suit::Hearts), (Suit::Clubs, Suit::Spades),
    (Suit::Diamonds, Suit::Clubs), (Suit::Diamonds, Suit::Diamonds), (Suit::Diamonds, Suit::Hearts), (Suit::Diamonds, Suit::Spades),
    (Suit::Hearts, Suit::Clubs), (Suit::Hearts, Suit::Diamonds), (Suit::Hearts, Suit::Hearts), (Suit::Hearts, Suit::Spades),
    (Suit::Spades, Suit::Clubs), (Suit::Spades, Suit::Diamonds), (Suit::Spades, Suit::Hearts), (Suit::Spades, Suit::Spades)
];

pub const RANK_PAIRS: [(Rank, Rank, bool); 169] = [
    (Rank::Ace, Rank::Ace, false), (Rank::Ace, Rank::King, true), (Rank::Ace, Rank::Queen, true), (Rank::Ace, Rank::Jack, true), (Rank::Ace, Rank::Ten, true), (Rank::Ace, Rank::Nine, true), (Rank::Ace, Rank::Eight, true), (Rank::Ace, Rank::Seven, true), (Rank::Ace, Rank::Six, true), (Rank::Ace, Rank::Five, true), (Rank::Ace, Rank::Four, true), (Rank::Ace, Rank::Three, true), (Rank::Ace, Rank::Two, true),
    (Rank::Ace, Rank::King, false), (Rank::King, Rank::King, false), (Rank::King, Rank::Queen, true), (Rank::King, Rank::Jack, true), (Rank::King, Rank::Ten, true), (Rank::King, Rank::Nine, true), (Rank::King, Rank::Eight, true), (Rank::King, Rank::Seven, true), (Rank::King, Rank::Six, true), (Rank::King, Rank::Five, true), (Rank::King, Rank::Four, true), (Rank::King, Rank::Three, true), (Rank::King, Rank::Two, true),
    (Rank::Ace, Rank::Queen, false), (Rank::King, Rank::Queen, false), (Rank::Queen, Rank::Queen, false), (Rank::Queen, Rank::Jack, true), (Rank::Queen, Rank::Ten, true), (Rank::Queen, Rank::Nine, true), (Rank::Queen, Rank::Eight, true), 
    (Rank::Queen, Rank::Seven, true), (Rank::Queen, Rank::Six, true), (Rank::Queen, Rank::Five, true), (Rank::Queen, Rank::Four, true), (Rank::Queen, Rank::Three, true), (Rank::Queen, Rank::Two, true),
    (Rank::Ace, Rank::Jack, false), (Rank::King, Rank::Jack, false), (Rank::Queen, Rank::Jack, false), (Rank::Jack, Rank::Jack, false), (Rank::Jack, Rank::Ten, true), (Rank::Jack, Rank::Nine, true), (Rank::Jack, Rank::Eight, true), (Rank::Jack, Rank::Seven, true), (Rank::Jack, Rank::Six, true), (Rank::Jack, Rank::Five, true), (Rank::Jack, Rank::Four, true), (Rank::Jack, Rank::Three, true), (Rank::Jack, Rank::Two, true),
    (Rank::Ace, Rank::Ten, false), (Rank::King, Rank::Ten, false), (Rank::Queen, Rank::Ten, false), (Rank::Jack, Rank::Ten, false), (Rank::Ten, Rank::Ten, false), (Rank::Ten, Rank::Nine, true), (Rank::Ten, Rank::Eight, true), (Rank::Ten, Rank::Seven, true), (Rank::Ten, Rank::Six, true), (Rank::Ten, Rank::Five, true), (Rank::Ten, Rank::Four, true), (Rank::Ten, Rank::Three, true), (Rank::Ten, Rank::Two, true),
    (Rank::Ace, Rank::Nine, false), (Rank::King, Rank::Nine, false), (Rank::Queen, Rank::Nine, false), (Rank::Jack, Rank::Nine, false), (Rank::Ten, Rank::Nine, false), (Rank::Nine, Rank::Nine, false), (Rank::Nine, Rank::Eight, true), (Rank::Nine, Rank::Seven, true), (Rank::Nine, Rank::Six, true), (Rank::Nine, Rank::Five, true), (Rank::Nine, Rank::Four, true), (Rank::Nine, Rank::Three, true), (Rank::Nine, Rank::Two, true),
    (Rank::Ace, Rank::Eight, false), (Rank::King, Rank::Eight, false), (Rank::Queen, Rank::Eight, false), (Rank::Jack, Rank::Eight, false), (Rank::Ten, Rank::Eight, false), (Rank::Nine, Rank::Eight, false), (Rank::Eight, Rank::Eight, false), (Rank::Eight, Rank::Seven, true), (Rank::Eight, Rank::Six, true), (Rank::Eight, Rank::Five, true), (Rank::Eight, Rank::Four, true), (Rank::Eight, Rank::Three, true), (Rank::Eight, Rank::Two, true),
    (Rank::Ace, Rank::Seven, false), (Rank::King, Rank::Seven, false), (Rank::Queen, Rank::Seven, false), (Rank::Jack, Rank::Seven, false), (Rank::Ten, Rank::Seven, false), (Rank::Nine, Rank::Seven, false), (Rank::Eight, Rank::Seven, false), (Rank::Seven, Rank::Seven, false), (Rank::Seven, Rank::Six, true), (Rank::Seven, Rank::Five, true), (Rank::Seven, Rank::Four, true), (Rank::Seven, Rank::Three, true), (Rank::Seven, Rank::Two, true),
    (Rank::Ace, Rank::Six, false), (Rank::King, Rank::Six, false), (Rank::Queen, Rank::Six, false), (Rank::Jack, Rank::Six, false), (Rank::Ten, Rank::Six, false), (Rank::Nine, Rank::Six, false), (Rank::Eight, Rank::Six, false), (Rank::Seven, Rank::Six, false), (Rank::Six, Rank::Six, false), (Rank::Six, Rank::Five, true), (Rank::Six, Rank::Four, true), (Rank::Six, Rank::Three, true), (Rank::Six, Rank::Two, true),
    (Rank::Ace, Rank::Five, false), (Rank::King, Rank::Five, false), (Rank::Queen, Rank::Five, false), (Rank::Jack, Rank::Five, false), (Rank::Ten, Rank::Five, false), (Rank::Nine, Rank::Five, false), (Rank::Eight, Rank::Five, false), (Rank::Seven, Rank::Five, false), (Rank::Six, Rank::Five, false), (Rank::Five, Rank::Five, false), (Rank::Five, Rank::Four, true), (Rank::Five, Rank::Three, true), (Rank::Five, Rank::Two, true),
    (Rank::Ace, Rank::Four, false), (Rank::King, Rank::Four, false), (Rank::Queen, Rank::Four, false), (Rank::Jack, Rank::Four, false), (Rank::Ten, Rank::Four, false), (Rank::Nine, Rank::Four, false), (Rank::Eight, Rank::Four, false), (Rank::Seven, Rank::Four, false), (Rank::Six, Rank::Four, false), (Rank::Five, Rank::Four, false), (Rank::Four, Rank::Four, false), (Rank::Four, Rank::Three, true), (Rank::Four, Rank::Two, true),
    (Rank::Ace, Rank::Three, false), (Rank::King, Rank::Three, false), (Rank::Queen, Rank::Three, false), (Rank::Jack, Rank::Three, false), (Rank::Ten, Rank::Three, false), (Rank::Nine, Rank::Three, false), (Rank::Eight, Rank::Three, false), (Rank::Seven, Rank::Three, false), (Rank::Six, Rank::Three, false), (Rank::Five, Rank::Three, false), (Rank::Four, Rank::Three, false), (Rank::Three, Rank::Three, false), (Rank::Three, Rank::Two, true),
    (Rank::Ace, Rank::Two, false), (Rank::King, Rank::Two, false), (Rank::Queen, Rank::Two, false), (Rank::Jack, Rank::Two, false), (Rank::Ten, Rank::Two, false), (Rank::Nine, Rank::Two, false), (Rank::Eight, Rank::Two, false), (Rank::Seven, Rank::Two, false), (Rank::Six, Rank::Two, false), (Rank::Five, Rank::Two, false), (Rank::Four, Rank::Two, false), (Rank::Three, Rank::Two, false), (Rank::Two, Rank::Two, false)
];

pub const PREFLOP_EQUITIES: [f32; 169] = [
    0.846525, 0.672725, 0.665925, 0.656225, 0.644975, 0.632675, 0.622225, 0.612325, 0.59565, 0.59305, 0.585225, 0.58695, 0.5722, 
    0.657075, 0.8256, 0.63255, 0.629575, 0.617225, 0.60675, 0.587325, 0.572875, 0.562175, 0.570125, 0.54455, 0.54225, 0.534675, 
    0.64045, 0.617, 0.800975, 0.600975, 0.592675, 0.580975, 0.55985, 0.5375, 0.534575, 0.523475, 0.52185, 0.50985, 0.50935, 
    0.636225, 0.603525, 0.587625, 0.7728, 0.57895, 0.561725, 0.536675, 0.52735, 0.5084, 0.4999, 0.48415, 0.482175, 0.477725, 
    0.623475, 0.6013, 0.576575, 0.55245, 0.74665, 0.539025, 0.526475, 0.5073, 0.4915, 0.471525, 0.469375, 0.45955, 0.450075, 
    0.606475, 0.573725, 0.551975, 0.53435, 0.517375, 0.7234, 0.50475, 0.48755, 0.4728, 0.451725, 0.43415, 0.43145, 0.428125, 
    0.596725, 0.55845, 0.534025, 0.51665, 0.49825, 0.4871, 0.692725, 0.481075, 0.458475, 0.437775, 0.43465, 0.4066, 0.407, 
    0.587, 0.557125, 0.519675, 0.498525, 0.472425, 0.462275, 0.4475, 0.66195, 0.454075, 0.4397, 0.417125, 0.40495, 0.38485, 
    0.5797, 0.5402, 0.5111, 0.47105, 0.45785, 0.446725, 0.4279, 0.4254, 0.635575, 0.429975, 0.4151, 0.393925, 0.382925, 0.57575, 
    0.526975, 0.5018, 0.472925, 0.44315, 0.423725, 0.412525, 0.40515, 0.399825, 0.59815, 0.416, 0.39745, 0.3787, 0.568475, 0.52985, 
    0.490975, 0.461875, 0.4279, 0.40555, 0.398975, 0.384775, 0.387625, 0.384525, 0.57145, 0.38915, 0.3719, 0.55695, 0.519575, 0.482225,
    0.448825, 0.4288, 0.393975, 0.377925, 0.374125, 0.359575, 0.359075, 0.351675, 0.54045, 0.364275, 0.549575, 0.50795, 0.471425, 
    0.440775, 0.4097, 0.389625, 0.377525, 0.34235, 0.345025, 0.337175, 0.33505, 0.3245, 0.506175
];

pub const PREFLOP_EQUITY_ADJUSTS: [f32; 169] = [
    1.1, 1.15, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.1, 1.15, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.1, 1.15, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.1, 1.15, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.9, 1.0, 1.0, 1.0, 1.1, 1.15, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.15, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.15, 1.1, 1.0, 1.0, 1.0, 1.0,
    0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.15, 1.1, 1.0, 1.0, 1.0,
    0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.15, 1.1, 1.0, 1.0,
    0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.15, 1.1, 1.0,
    1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.15, 1.1,
    1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.15,
    1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1,
];


fn parse_cards(names: &str) -> Vec<Card> {
    names
        .split_ascii_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

fn test_eval() {
    let cards = parse_cards("2H 3D 5S 9C KD");
    let hole_cards = parse_cards("2C 3H");
    let hole_cards2 = parse_cards("9C 3H");
    let (score, perm) = eval7([cards.clone(), hole_cards].concat());
    let (score2, perm2) = eval7([cards, hole_cards2].concat());
    println!("{}", score);
    println!("{}", score2);
}

fn test_equity() {
    let ai = AI::new(String::from("AI1"), parse_cards("7D 4C"));
    let equity = ai.equity(10000, &[]);
    println!("Equity: {}", equity);
}

fn gen_rank_pair_table() {
    let names = ["Ace", "King", "Queen", "Jack", "Ten", "Nine", "Eight", "Seven", "Six", "Five", "Four", "Three", "Two"];
    for i in 0..13 {
        for j in 0..13 {
            let suited = i < j;
            print!("(Rank::{}, Rank::{}, {}), ", names[i.min(j)], names[i.max(j)], suited);
        }
        println!();
    }
}

fn gen_preflop_equities() {
    let mut ai = AI::new(String::from("AI1"), vec![DECK[0], DECK[1]]);
    for rank_pair in RANK_PAIRS {
        let card1 = Card::new(rank_pair.0, Suit::Spades);
        let card2 = Card::new(rank_pair.1, if rank_pair.2 { Suit::Spades } else { Suit::Hearts });
        ai.set_hole_cards(vec![card1, card2]);
        let equity = ai.equity(20000, &[]);
        print!("{}, ", equity);
    }
}

fn gen_preflop_equity_adjusts() {    
    for i in 0..13i8 {
        for j in 0..13i8 {
            let suited = i < j;
            let mut adj = 1.0;
            if suited && (i-j).abs() == 1 {
                adj = 1.2;
            } else if suited && (i-j).abs() == 2 {
                adj = 1.1;
            } else if !suited && (i-j).abs() == 0 {
                adj = 1.1;
            } else if !suited && (i-j).abs() > 3 {
                adj = 0.8;
            }
            print!("{:.1}, ", adj);
        }
        println!();
    }
}

pub fn print_pair_range<F>(bg: F, schm: u8)
    where F: Fn(usize) -> u8,
{    
    println!();
    for i in 0..13u8 {
        for j in 0..13 {
            let suited = i < j;
            let bg = bg((i*13+j) as usize).clamp(0, 5);
            let (ri, rj) = if suited { (14-i, 14-j) } else { (14-j, 14-i) };
            let (f1, f2) = match suited {
                true => ("\x1b[90m", "\x1b[90m"),
                false => ("\x1b[90m", "\x1b[91m"),
            };  

            let b1 = format!("\x1b[48;5;{}m", schm + 36 * (5 - bg));

            print!("{}{}{}{}{}\x1b[0m", b1, f1, Rank::try_from(ri).unwrap(), f2, Rank::try_from(rj).unwrap());
        }
        println!();
    }
    println!();
}