use crate::value::{ValueRef, val};

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add,
    Mul,
    Exp,
    Pow(f64),
    ReLU,
}

impl Op {
    pub fn apply_backward(&self, node_grad: f64, node_data: f64, parents: &[ValueRef]) {
        match self {
            Op::Add => {
                for parent in parents {
                    parent.borrow_mut().grad += node_grad;
                }
            },
            Op::Mul => {
                let a = parents[0].borrow().data;
                let b = parents[1].borrow().data;
                parents[0].borrow_mut().grad += b * node_grad;
                parents[1].borrow_mut().grad += a * node_grad;
            },
            Op::Exp => {
                let grad = node_data * node_grad;
                parents[0].borrow_mut().grad += grad;
            },
            Op::Pow(exponent) => {
                let base = parents[0].borrow().data;
                let grad = exponent * base.powf(exponent - 1.0) * node_grad;
                parents[0].borrow_mut().grad += grad;
            },
            Op::ReLU => {
                let grad = if node_data > 0.0 { node_grad } else { 0.0 };
                parents[0].borrow_mut().grad += grad;
            },
        }
    }
}

pub fn add(a: &ValueRef, b: &ValueRef) -> ValueRef {
    let out = val(a.borrow().data + b.borrow().data);
    {
        let mut o = out.borrow_mut();
        o.op = Some(Op::Add);
        o.parents = vec![a.clone(), b.clone()];
    }
    out
}

pub fn mul(a: &ValueRef, b: &ValueRef) -> ValueRef {
    let out = val(a.borrow().data * b.borrow().data);
    {
        let mut o = out.borrow_mut();
        o.op = Some(Op::Mul);
        o.parents = vec![a.clone(), b.clone()];
    }
    out
}

pub fn relu(a: &ValueRef) -> ValueRef {
    let out = val(if a.borrow().data > 0.0 { a.borrow().data } else { 0.0 });
    {
        let mut o = out.borrow_mut();
        o.op = Some(Op::ReLU);
        o.parents = vec![a.clone()];
    }
    out
}

pub fn neg(a: &ValueRef) -> ValueRef {
    let out = val(-a.borrow().data);
    {
        let mut o = out.borrow_mut();
        o.op = Some(Op::Mul);
        o.parents = vec![a.clone(), val(-1.0)];
    }
    out
}

pub fn sub(a: &ValueRef, b: &ValueRef) -> ValueRef {
    add(a, &neg(b))
}

pub fn div(a: &ValueRef, b: &ValueRef) -> ValueRef {
    mul(a, &pow(b, -1.0))
}

pub fn pow(a: &ValueRef, exponent: f64) -> ValueRef {
    let out = val(a.borrow().data.powf(exponent));
    {
        let mut o = out.borrow_mut();
        o.op = Some(Op::Pow(exponent));
        o.parents = vec![a.clone()];
    }
    out
}

pub fn exp(a: &ValueRef) -> ValueRef {
    let out = val(a.borrow().data.exp());
    {
        let mut o = out.borrow_mut();
        o.op = Some(Op::Exp);
        o.parents = vec![a.clone()];
    }
    out
}
