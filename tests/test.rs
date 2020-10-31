
use read_pdf_417::decode_image;

#[test]
fn text() {
    let buffer = include_bytes!("./assets/text.gif");
    let img = image::load_from_memory(buffer).unwrap();

    let detected = decode_image(&img);

    dbg!(&detected);
    assert!(detected.is_some());
}

#[test]
fn easy() {
    let buffer = include_bytes!("./assets/easy.png");
    let img = image::load_from_memory(buffer).unwrap();

    let detected = decode_image(&img);

    dbg!(&detected);
    assert!(detected.is_some());
}

#[test]
fn easy2() {
    let buffer = include_bytes!("./assets/easy2.png");
    let img = image::load_from_memory(buffer).unwrap();

    let detected = decode_image(&img);

    dbg!(&detected);
    assert!(detected.is_some());
}
// #[test]
// fn id() {
//     let buffer = include_bytes!("./assets/id.png");
//     let img = image::load_from_memory(buffer).unwrap();

//     let detected = decode_image(&img);

//     dbg!(&detected);
//     assert!(detected.is_some());
    
// }