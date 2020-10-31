#![feature(iter_order_by)]
#![feature(const_generics)]
use image::{DynamicImage, GenericImageView};
use std::collections::BTreeMap;

mod lookup_table;

pub const START: Pattern<8> = Pattern::new([8, 1, 1, 1, 1, 1, 1, 3], 17);
// pub const START: [bool; 17] = [true, true, true, true, true, true, true, true, false, true, false, true, false, true, false, false, false];
pub const STOP: Pattern<9> = Pattern::new([7, 1, 1, 3, 1, 1, 1, 2, 1], 18);
// pub const STOP: [bool; 18] = [true, true, true, true, true, true, true, false, true, false, false, false, true, false, true, false, false, true];

pub type Point = (u32, u32);


struct RowIndicator {
    row_number: usize,
    num_rows: usize,
    num_columns: usize,
    error_correction_level: usize
}

#[derive(PartialEq, Eq)]
enum RowIndicatorType {
    Left,
    Right
}

#[derive(Debug)]
pub struct BoundingBox {
    start: Point,
    end: Point,
}

#[derive(Debug)]
pub struct Detection {
    position: BoundingBox,
    scale: usize,
}

pub struct Pattern<const SIZE: usize> {
    counts: [usize; SIZE],
    sum: usize,
}

impl<const SIZE: usize> Pattern<SIZE> {
    pub const fn new(counts: [usize; SIZE], sum: usize) -> Self {
        Self { counts, sum }
    }
    pub fn matches<'a, I>(&self, array: &mut I, scale: usize, allowed_error: usize) -> bool
    where
        I: Iterator<Item = &'a usize> + std::fmt::Debug,
    {
        self.counts.iter().eq_by(array, |&a, &b| {
            let target = b / scale;
            std::cmp::min(0, target - allowed_error) <= a
                && a <= std::cmp::max(usize::MAX, target + allowed_error)
        })
    }
    pub fn sum(&self) -> usize {
        self.sum
    }
}

// fn find_pattern(buffer: &[bool], pattern: &[u8]) -> bool {

//     // Store the pattern so far and where we are at in analyizing it
//     let mut counter = [1, 0, 0, 0, 0, 0, 0, 0];
//     let mut index = 0;

//     for pixel in buffer {
//         // dbg!(start, y, pixel);

//         // Color changed
//         if (*pixel && (index % 2 != 0)) || (!pixel && (index % 2 == 0)) {
//             // dbg!("Color change!", index % 2);

//             // Check if the pattern matches so far
//             let pattern_matches = PDF417::START[..index] == counter[..index];
//             if !pattern_matches {
//                 break;
//             }

//             index += 1;
//         }

//         // Index is larger than the pattern we are trying to match, time to quit
//         if index + 1 > pattern.len() {
//             // dbg!("Done searching!");
//             let pattern_matches = PDF417::START == counter;
//             if pattern_matches {
//                 return true;
//             }

//             break;
//         }

//         counter[index] += 1;
//     }

//     false
// }

pub fn detect_barcode(buffer: &image::GrayImage, allowed_error: usize) -> Option<Detection> {
    let mut start_pos: Option<Point> = None;
    let mut stop_pos: Option<Point> = None;
    let mut scale = 1;

    // Iterate through every pixel looking for a start pattern
    'search: for y in 0..buffer.height() {
        // let row_end_index = (y + 1) * width - 1;
        let mut start_counters: RingBuffer<8> = RingBuffer::new();
        let mut stop_counters: RingBuffer<9> = RingBuffer::new();
        let mut prev = false;
        for x in 0..buffer.width() {
            // Get pixel value
            let pixel = buffer.get_pixel(x, y)[0];

            let curr = if pixel < 127 { true } else { false };

            // If color change, move to next element in counters
            if prev != curr {
                start_counters.push_back(1);
                stop_counters.push_back(1);
            } else {
                start_counters.inc_back(1);
                stop_counters.inc_back(1);
            }

            let stop_error = stop_counters.sum() % STOP.sum();
            let start_error = start_counters.sum() % START.sum();

            // Look for start pattern
            if start_error <= allowed_error
                && start_counters[0] != 0
                && start_counters[0] % START.counts[0] == 0
            {
                // println!("[{:?}] {} <= {}", start_counters.sum(), start_error, allowed_error);
                scale = start_counters.sum() / START.sum();
                let matches_pattern =
                    { START.matches(&mut start_counters.iter(), scale, allowed_error) };

                if matches_pattern {
                    // println!("FOUND START");
                    start_pos = Some((x - (start_counters.sum() - 1) as u32, y));
                }
            }

            // Look for stop pattern
            if start_pos.is_some()
                && stop_error <= allowed_error
                && stop_counters[0] != 0
                && stop_counters[0] % STOP.counts[0] == 0
            {
                // println!("[{:?}] {} <= {}", stop_counters.sum(), stop_error, allowed_error);
                let matches_pattern =
                    { STOP.matches(&mut stop_counters.iter(), scale, allowed_error) };

                if matches_pattern {

                    // Keep going down until the black stops
                    // PDF417 has a quiet zone around the barcode
                    let mut black_stop = y;
                    while black_stop + 1 < buffer.height() && buffer.get_pixel(x, black_stop + 1)[0] < 127 {
                        black_stop += 1;
                    }
                    // println!("FOUND STOP");
                    stop_pos = Some((x - stop_counters.sum() as u32, black_stop));
                }
            }

            if start_pos.is_some() && stop_pos.is_some() {
                break 'search;
            }

            prev = curr;
        }

        // Reset start_pos because we did not find a stop pattern on this row
        start_pos = None;
    }

    // If either a start or a stop wasn't found, return nothing
    if start_pos.is_none() || stop_pos.is_none() {
        return None;
    }

    Some(Detection {
        position: BoundingBox {
            start: start_pos.unwrap(),
            end: stop_pos.unwrap(),
        },
        scale,
    })
}

pub fn decode_image(img: &DynamicImage) -> Option<Vec<String>> {
    // The dimensions method returns the images width and height.
    let (width, height) = img.dimensions();
    println!("Dimensions: {}x{}", width, height);

    let mut grey = img.grayscale();

    // let bw: Vec<bool> = luma.as_raw().iter().map(|luma| {
    //     match *luma > 127 {
    //         true => false,
    //         false => true
    //     }
    // })
    //     .collect();

    let mut detection = None;
    let mut i = 0;

    while detection.is_none() && i < 4 {
        println!("Rotation {}", i * 90);

        grey.save(format!("/tmp/grey-{}.png", i * 90))
            .expect("Save file");

        let luma = grey.as_luma8().unwrap();

        detection = detect_barcode(luma, 0);

        if detection.is_none() {
            grey = grey.rotate90();
        }

        i += 1;
    }

    dbg!(&detection);

    if detection.is_none() {
        return None;
    }
    let detection = detection.unwrap();
    let mut decoder = Decoder::new(grey.as_luma8().unwrap(), detection.scale, detection.position);

    decoder.decode_barcode();

    // let barcode = barcode.unwrap();
    // let scale = barcode.scale as u32;

    // let cropped = grey.crop_imm(barcode.position.start.0, barcode.position.start.1, barcode.position.end.0 - barcode.position.start.0 + STOP.len() as u32, height);
    // let small = cropped.resize(width / scale, height / scale, image::imageops::FilterType::CatmullRom);

    // let small_luma = small.as_luma8().unwrap();

    // small_luma.save("/tmp/grey.png").expect("Save file");

    None
}

#[derive(Copy, Clone)]
enum Command {
    LatchToAlpha,
    ShiftToAlpha,
    LatchToLower,
    LatchToMixed,
    LatchToPunctuation,
    ShiftToPunctuation,
}

#[derive(Copy, Clone)]
enum Value {
    Character(char),
    Command(Command),
}

const ALPHA: [Value; 30] = [
    Value::Character('A'),
    Value::Character('B'),
    Value::Character('C'),
    Value::Character('D'),
    Value::Character('E'),
    Value::Character('F'),
    Value::Character('G'),
    Value::Character('H'),
    Value::Character('I'),
    Value::Character('J'),
    Value::Character('K'),
    Value::Character('L'),
    Value::Character('M'),
    Value::Character('N'),
    Value::Character('O'),
    Value::Character('P'),
    Value::Character('Q'),
    Value::Character('R'),
    Value::Character('S'),
    Value::Character('T'),
    Value::Character('U'),
    Value::Character('V'),
    Value::Character('W'),
    Value::Character('X'),
    Value::Character('Y'),
    Value::Character('Z'),
    Value::Character(' '),
    Value::Command(Command::LatchToLower),
    Value::Command(Command::LatchToMixed),
    Value::Command(Command::ShiftToPunctuation),
];
const LOWER: [Value; 30] = [
    Value::Character('a'),
    Value::Character('b'),
    Value::Character('c'),
    Value::Character('d'),
    Value::Character('e'),
    Value::Character('f'),
    Value::Character('g'),
    Value::Character('h'),
    Value::Character('i'),
    Value::Character('j'),
    Value::Character('k'),
    Value::Character('l'),
    Value::Character('m'),
    Value::Character('n'),
    Value::Character('o'),
    Value::Character('p'),
    Value::Character('q'),
    Value::Character('r'),
    Value::Character('s'),
    Value::Character('t'),
    Value::Character('u'),
    Value::Character('v'),
    Value::Character('w'),
    Value::Character('x'),
    Value::Character('y'),
    Value::Character('z'),
    Value::Character(' '),
    Value::Command(Command::ShiftToAlpha),
    Value::Command(Command::LatchToMixed),
    Value::Command(Command::ShiftToPunctuation),
];
const MIXED: [Value; 30] = [
    Value::Character('0'),
    Value::Character('1'),
    Value::Character('2'),
    Value::Character('3'),
    Value::Character('4'),
    Value::Character('5'),
    Value::Character('6'),
    Value::Character('7'),
    Value::Character('8'),
    Value::Character('9'),
    Value::Character('&'),
    Value::Character('\r'), // CR
    Value::Character('\t'), // HT
    Value::Character(','),
    Value::Character(':'),
    Value::Character('#'),
    Value::Character('-'),
    Value::Character('.'),
    Value::Character('$'),
    Value::Character('/'),
    Value::Character('+'),
    Value::Character('%'),
    Value::Character('*'),
    Value::Character('='),
    Value::Character('^'),
    Value::Command(Command::LatchToPunctuation),
    Value::Character(' '),
    Value::Command(Command::LatchToLower),
    Value::Command(Command::LatchToMixed),
    Value::Command(Command::ShiftToPunctuation),
];
const PUNCTUATION: [Value; 30] = [
    Value::Character(';'),
    Value::Character('<'),
    Value::Character('>'),
    Value::Character('@'),
    Value::Character('['),
    Value::Character('\\'),
    Value::Character(']'),
    Value::Character('_'),
    Value::Character('\''),
    Value::Character('~'),
    Value::Character('!'),
    Value::Character('\r'), // CR
    Value::Character('\t'), // HT
    Value::Character(','),
    Value::Character(':'),
    Value::Character('\n'), // LF
    Value::Character('-'),
    Value::Character('.'),
    Value::Character('$'),
    Value::Character('/'),
    Value::Character('"'),
    Value::Character('|'),
    Value::Character('*'),
    Value::Character('('),
    Value::Character(')'),
    Value::Character('?'),
    Value::Character('{'),
    Value::Character('}'),
    Value::Character('\''),
    Value::Command(Command::LatchToAlpha),
];

#[derive(Copy, Clone)]
enum Mode {
    Text(SubModeText),
    Byte(SubModeByte),
    Numeric
}
#[derive(Copy, Clone)]
enum SubModeText {
    Alpha,
    Lower,
    Mixed,
    Punctuation,
}

#[derive(Copy, Clone)]
enum SubModeByte {
    // total number of bytes to be encoded is NOT an integer multiple of 6
    Normal,
    // total number of bytes to be encoded is an integer multiple of 6
    Six,
}

struct Decoder<'a> {
    lookup_table: BTreeMap<[u8; 8], usize>,
    counters: RingBuffer<8>,
    buffer: &'a image::GrayImage,
    // Used to determine mode through latches
    mode: Mode,
    // Used for Shifts
    temp_mode: Option<Mode>,
    scale: usize,
    bounds: BoundingBox
}
impl<'a> Decoder<'a> {
    pub fn new(buffer: &'a image::GrayImage, scale: usize, bounds: BoundingBox) -> Self {
        let mut lookup_table = BTreeMap::new();
        for (code, pattern) in lookup_table::VALUES.iter() {
            lookup_table.insert(*pattern, *code);
        }
        Decoder {
            lookup_table,
            counters: RingBuffer::new(),
            buffer,
            mode: Mode::Text(SubModeText::Alpha),
            temp_mode: None,
            scale,
            bounds
        }
    }
    // pub fn cluster_number<const SIZE: usize>(buffer: &RingBuffer<SIZE>, scale: usize) -> usize {
    //     let T1 = (buffer[0] + buffer[1]) * scale;
    //     let T2 = (buffer[1] + buffer[2]) * scale;
    //     let T5 = (buffer[4] + buffer[5]) * scale;
    //     let T6 = (buffer[5] + buffer[6]) * scale;
    //     (T1 - T2 + T5 - T6 + 9) % 9
    // }

    fn process_value(&mut self, idx: usize, mode: Mode) -> Option<char> {
        match mode {
            Mode::Text(sub_mode) => {
                let value = match sub_mode {
                    SubModeText::Alpha => ALPHA[idx],
                    SubModeText::Lower => LOWER[idx],
                    SubModeText::Mixed => MIXED[idx],
                    SubModeText::Punctuation => PUNCTUATION[idx]
                };
                match value {
                    Value::Command(cmd) => {
                        match cmd {
                            Command::LatchToAlpha => self.mode = Mode::Text(SubModeText::Alpha),
                            Command::ShiftToAlpha => self.temp_mode = Some(Mode::Text(SubModeText::Alpha)),
                            Command::LatchToLower => self.mode = Mode::Text(SubModeText::Lower),
                            Command::LatchToMixed => self.mode = Mode::Text(SubModeText::Mixed),
                            Command::LatchToPunctuation => self.mode = Mode::Text(SubModeText::Punctuation),
                            Command::ShiftToPunctuation => self.temp_mode = Some(Mode::Text(SubModeText::Punctuation)),
                        };
                        None
                    },
                    Value::Character(c) => Some(c)
                }
            },
            _ => unimplemented!()
        }
    }

    fn decode_chunk(&mut self, (value1, value2): (usize, usize)) -> (Option<char>, Option<char>) {
        let mode = self.temp_mode.take().unwrap_or(self.mode);
        let high = self.process_value(value1, mode);
        let low = self.process_value(value2, mode);
        (high, low)
    }

    fn decode_codeword(codeword: usize) -> (usize, usize) {
        assert!(codeword < 900, "Cannot decode a codeword over 900");
        let second = codeword % 30;
        let first = (codeword - second) / 30;
        (first, second)
    }

    fn read_codeword(&self, x: u32, y: u32) -> Option<&usize> {
        let start = (self.buffer.width() * y + x) as usize;

        // Optimize: Don't use a vector, use an array
        let container = Vec::with_capacity(8);

        // Codewords are always 17 modules long
        let length = 17 * self.scale;
        let end = start + length;
        let segment = &self.buffer.as_raw()[start..end]
            .iter()
            .step_by(self.scale)
            .map(|&pixel| if pixel < 127 { true } else { false })
            .fold(container, |mut acc, black| {
                if black && acc.len() % 2 == 0 {
                    acc.push(1);
                } else if !black && acc.len() % 2 == 1 {
                    acc.push(1);
                } else {
                    let idx = acc.len() - 1;
                    acc[idx] += 1;
                }

                acc
            });

        let key = segment.as_slice();
        let code = self.lookup_table.get(key);
        
        code
    }

    fn read_row_indicator(&self, x: u32, y: u32, _type: RowIndicatorType) -> RowIndicator /*-> Result<RowIndicator, Box<dyn std::error::Error>>*/ {
        let i1 = self.read_codeword(x, y).unwrap();
        let i2 = self.read_codeword(x, y + self.scale as u32).unwrap();
        let i3 = self.read_codeword(x, y + (self.scale * 2) as u32).unwrap();

        // TODO: Make sure i(x) / 30 is equal
        // In other words, check that you are always reading the correct row 
        let row_number = i1 / 30 + 1;
        let indicator = match _type {
            RowIndicatorType::Left => {
                let num_rows = i1 % 30 + 1;
                let error_correction_level = i2 % 30;
                let num_columns = i3 % 30 + 1;
                RowIndicator {
                    row_number,
                    num_rows,
                    num_columns,
                    error_correction_level
                }
            },
            RowIndicatorType::Right => {
                let num_rows = i2 % 30 + 1;
                let error_correction_level = i3 % 30;
                let num_columns = i1 % 30 + 1;
                RowIndicator {
                    row_number,
                    num_rows,
                    num_columns,
                    error_correction_level
                }
            }
        };

        debug_assert!(row_number >= 1 && row_number <= 90, "row indicator row_number out of allowed range");
        debug_assert!(indicator.num_rows >= 3 && indicator.num_rows <= 90, "row indicator num_rows out of allowed range");
        debug_assert!(indicator.num_columns >= 1 && indicator.num_columns <= 30, "row indicator num_columns out of allowed range");
        debug_assert!(indicator.error_correction_level <= 8, "row indicator error_correction_level out of allowed range");

        indicator
    }

    fn run_special(&mut self, codeword: usize) {
        match codeword {
            // Latch to Text
            900 => self.mode = Mode::Text(SubModeText::Alpha),
            901 => self.mode = Mode::Byte(SubModeByte::Normal),
            902 => self.mode = Mode::Numeric,
            913 => self.temp_mode = Some(Mode::Byte(SubModeByte::Normal)),
            924 => self.mode = Mode::Byte(SubModeByte::Six),
            _ => unimplemented!()
        }
    }

    fn read_row(&mut self, x: u32, y: u32, width: usize) -> String {
        let num_codewords = width / 17;
        let mut output = String::with_capacity(num_codewords);
        let mut start = x as usize;
        let end = start + width;

        

        while start < end {
            let codeword = self.read_codeword(start as u32, y);
            let codeword = codeword.expect("No codeword matching").to_owned();

            // Codewords 900 and above are special commands
            if codeword < 900 {
                dbg!(codeword);
                let chunk = Self::decode_codeword(codeword);
                dbg!(chunk);
                let value = self.decode_chunk(chunk);
                dbg!(value);
                value.0.map(|v| output.push(v));
                value.1.map(|v| output.push(v));
            } else {
                self.run_special(codeword);
            }

            start += 17;
        }

        output
    }

    pub fn decode_barcode(&mut self) {
        let start_y = self.bounds.start.1;
        let start_x = self.bounds.start.0 + (START.sum * self.scale) as u32;
        let start_x = self.bounds.start.0 + (START.sum * self.scale) as u32;
        let end_x = self.bounds.end.0;
        let width = (end_x - start_x) as usize;

        // Iterate through every pixel looking for a start pattern
        'search: for y in (start_y..self.buffer.height()).step_by(self.scale * 3) {
            let start_identifier = self.read_row_indicator(start_x, y, RowIndicatorType::Left);
            let row_text = self.read_row(start_x, y, width);
            dbg!(y, row_text);
        }
    }
}

#[derive(Debug)]
struct RingBuffer<const SIZE: usize> {
    buffer: [usize; SIZE],
    // Points to the end of the array
    pointer: usize,
    sum: usize,
}

impl<const MAX_SIZE: usize> RingBuffer<MAX_SIZE> {
    const LAST_INDEX: usize = MAX_SIZE - 1;
    pub fn new() -> Self {
        RingBuffer {
            buffer: [0usize; MAX_SIZE],
            pointer: 0,
            sum: 0,
        }
    }
    pub fn push_back(&mut self, value: usize) {
        if self.pointer + 1 >= MAX_SIZE {
            self.pointer = 0;
        } else {
            self.pointer += 1;
        }
        self.sum -= self[Self::LAST_INDEX];
        self[Self::LAST_INDEX] = value;
        self.sum += value;
    }
    pub fn inc_back(&mut self, by: usize) -> usize {
        self[Self::LAST_INDEX] += by;
        self.sum += by;
        self[Self::LAST_INDEX]
    }
    pub fn last(&self) -> usize {
        self[Self::LAST_INDEX]
    }
    pub fn first(&self) -> usize {
        self[0]
    }
    pub fn sum(&self) -> usize {
        self.sum
    }
    pub fn into_inner(self) -> [usize; MAX_SIZE] {
        self.buffer
    }
    pub fn iter(
        &self,
    ) -> std::iter::Chain<std::slice::Iter<'_, usize>, std::slice::Iter<'_, usize>> {
        self.buffer[self.pointer..]
            .iter()
            .chain(self.buffer[..self.pointer].iter())
    }
    pub fn reset(&mut self) {
        self.buffer = [0usize; MAX_SIZE];
    }
}

// impl<const MAX_SIZE: usize> std::ops::Deref for RingBuffer<MAX_SIZE> {
//     type Target = &[usize; MAX_SIZE];

//     fn deref(&self) -> &Self::Target {
//         self.buffer
//     }
// }

impl<const MAX_SIZE: usize> std::ops::Index<usize> for RingBuffer<MAX_SIZE> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        let idx = (index + self.pointer) % MAX_SIZE;
        &self.buffer[idx]
    }
}

impl<const MAX_SIZE: usize> std::ops::IndexMut<usize> for RingBuffer<MAX_SIZE> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let idx = (index + self.pointer) % MAX_SIZE;
        &mut self.buffer[idx]
    }
}

#[test]
fn ringbuffer() {
    let mut buffer: RingBuffer<4> = RingBuffer::new();
    buffer.push_back(1);
    assert_eq!(buffer.last(), 1);
    assert_eq!(buffer.buffer, [1, 0, 0, 0]);

    buffer.inc_back(2);
    assert_eq!(buffer.sum(), 3);
    assert_eq!(buffer.last(), 3);
    assert_eq!(buffer.buffer, [3, 0, 0, 0]);

    buffer.push_back(3);
    assert_eq!(buffer.last(), 3);
    assert_eq!(buffer.buffer, [3, 3, 0, 0]);

    buffer.push_back(4);
    assert_eq!(buffer.last(), 4);
    assert_eq!(buffer.buffer, [3, 3, 4, 0]);

    buffer.push_back(5);
    assert_eq!(buffer.last(), 5);
    assert_eq!(buffer.buffer, [3, 3, 4, 5]);

    buffer.push_back(6);
    assert_eq!(buffer.last(), 6);
    assert_eq!(buffer.buffer, [6, 3, 4, 5]);
    assert_eq!(buffer.first(), 3);

    assert_eq!(buffer.sum(), 18);
    assert!(buffer.iter().eq([3, 4, 5, 6].iter()));

    buffer.push_back(7);
    assert_eq!(buffer.last(), 7);
    assert_eq!(buffer.buffer, [6, 7, 4, 5]);
    assert_eq!(buffer.first(), 4);

    assert_eq!(buffer.sum(), 22);
    assert!(buffer.iter().eq([4, 5, 6, 7].iter()));
}
