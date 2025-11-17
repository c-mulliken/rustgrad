use rustgrad::{add, backward, mul, val};

fn main() {
    // x = 2.0, y = 3.0
    let x = val(2.0);
    let y = val(3.0);

    // z = (x + y) * y
    let q = add(&x, &y);
    let z = mul(&q, &y);

    println!("x.data = {}", x.borrow().data);
    println!("y.data = {}", y.borrow().data);
    println!("z.data = (x + y) * y = {}", z.borrow().data);

    backward(&z);

    println!("dz/dx (x.grad) = {}", x.borrow().grad);
    println!("dz/dy (y.grad) = {}", y.borrow().grad);
}
