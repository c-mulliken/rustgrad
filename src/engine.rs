use std::rc::Rc;
use crate::value::{ValueRef, Value};

fn visit(node: &ValueRef, visited: &mut std::collections::HashSet<*const std::cell::RefCell<Value>>, order: &mut Vec<ValueRef>) {
    let ptr = Rc::as_ptr(node);
    if visited.contains(&ptr) {
        return;
    } else {
        visited.insert(ptr);
    }

    for parent in &node.borrow().parents {
        visit(parent, visited, order);
    }
    order.push(node.clone());
}

pub fn topo_sort(root: &ValueRef) -> Vec<ValueRef> {
    let mut visited = std::collections::HashSet::new();
    let mut order = Vec::new();   
    visit(root, &mut visited, &mut order);
    order
}

pub fn backward(root: &ValueRef) {
    let mut order = topo_sort(root);
    root.borrow_mut().grad = 1.0;
    order.reverse();

    for node in order {
        let (op, node_grad, node_data, parents) = {
            let n = node.borrow();
            let op = n.op;
            let node_grad = n.grad;
            let node_data = n.data;
            let parents = n.parents.clone();
            (op, node_grad, node_data, parents)
        };

        if let Some(op) = op {
            op.apply_backward(node_grad, node_data, &parents);
        }
    }
}

pub fn zero_grad(root: &ValueRef) {
    let order = topo_sort(root);
    for node in order {
        node.borrow_mut().grad = 0.0;
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{add, mul, exp, pow, relu};
    use crate::value::val;
    use crate::utils::approx;

    #[test]
    fn test_topo_sort() {
        let x = val(2.0);
        let y = val(3.0);
        let z = add(&x, &y);
        let w = mul(&z, &x);

        let order = topo_sort(&w);
        let order_ptrs: Vec<*const std::cell::RefCell<Value>> = order.iter().map(|v| Rc::as_ptr(v)).collect();

        assert!(order_ptrs.contains(&Rc::as_ptr(&x)));
        assert!(order_ptrs.contains(&Rc::as_ptr(&y)));
        assert!(order_ptrs.contains(&Rc::as_ptr(&z)));
        assert_eq!(order_ptrs.last().unwrap(), &Rc::as_ptr(&w));
    }

    #[test]
    fn test_backward() {
        let x = val(2.0);
        let y = pow(&x, 3.0);
        let z = exp(&y);

        backward(&z);

        let x_grad = x.borrow().grad;
        let expected_grad = 3.0 * 2.0_f64.powf(2.0) * (2.0_f64.powf(3.0)).exp();

        assert!(approx(x_grad, expected_grad, 1e-6));
    }

    #[test]
    fn test_relu_backward() {
        let x = val(-1.0);
        let y = val(3.0);
        let z = add(&x, &y);
        let w = relu(&z);

        backward(&w);

        let x_grad = x.borrow().grad;
        let y_grad = y.borrow().grad;

        assert!(approx(x_grad, 1.0, 1e-6));
        assert!(approx(y_grad, 1.0, 1e-6));
    }

    #[test]
    fn test_zero_grad() {
        let x = val(2.0);
        let y = mul(&x, &x);
        let z = add(&y, &x);

        backward(&z);
        zero_grad(&z);
        let x_grad = x.borrow().grad;
        let y_grad = y.borrow().grad;
        assert!(approx(x_grad, 0.0, 1e-6));
        assert!(approx(y_grad, 0.0, 1e-6));
    }
}
