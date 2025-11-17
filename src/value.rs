use std::cell::RefCell;
use std::rc::Rc;
use crate::ops::Op;

pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub op: Option<Op>,
    pub parents: Vec<ValueRef>,
}

pub type ValueRef = Rc<RefCell<Value>>;

pub fn val(x: f64) -> ValueRef {
    Rc::new(RefCell::new(Value {
        data: x,
        grad: 0.0,
        op: None,
        parents: Vec::new(),
    }))
}


