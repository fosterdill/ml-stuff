(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){

let hidden = 15;
let inputs = 2;
let outputs = 3;
var a2ones = Matrix.ones(1, 1);
a2shapeones = Matrix.ones(hidden + 1, 1);

const sigmoid = value => {
  return (1 / (1 + Math.exp(-value)));
};

const addRowToTop = (m, r) => {
  return Matrix.augment(r.T, m.T).T;
};

const getColumn = (m, j) => {
  let result = Matrix.zeros(m.shape[0], 1);

  for (var i = 0; i < m.shape[0]; i++) {
    result.set(i, 0, m.get(i, j));
  }

  return result;
};

const forwardPropagate = (theta1, theta2, inputs, x) => {
  let a1 = x;
  let a2 = Matrix.multiply(theta1, a1).map(sigmoid);
  a2 = addRowToTop(a2, a2ones);
  let a3 = Matrix.multiply(theta2, a2).map(sigmoid);

  return {
    a1,
    a2,
    a3 
  };
};

const costFunction = (X, y, theta1, theta2, inputs, outputs, hidden) => {
  let m = X.shape[1];
  let n = X.shape[0];
  let bigDelta1 = Matrix.zeros(hidden + 1, inputs + 1);
  let bigDelta2 = Matrix.zeros(outputs, hidden + 1);
  // let a3s = [];
  X = addRowToTop(X, Matrix.ones(1, m));

  for (let i = 0; i < m; i++) {
    let {
      a1, 
      a2, 
      a3
    } = forwardPropagate(
      theta1, 
      theta2, 
      inputs,
      getColumn(X, i)
    );

    // a3s[i] = math.squeeze(a3);
    yi = getColumn(y, i);
    let d3 = Matrix.subtract(a3, yi);
    let d2 = Matrix.multiply(theta2.T, d3)
      .product(a2)
      .product(
        Matrix.subtract(a2shapeones, a2)
      );

    bigDelta1.add(Matrix.multiply(d2, a1.T));
    bigDelta2.add(Matrix.multiply(d3, a2.T));
  }
  let D1 = bigDelta1.scale(1 / m);
  let D2 = bigDelta2.scale(1 / m);

  // let J = math.multiply(
  //   -(1/m), 
  //   math.add(
  //     math.multiply(
  //       y, 
  //       math.log(a3s)
  //     ), 
  //     math.multiply(
  //       math.subtract(1, y), 
  //       math.log(math.subtract(1, a3s))
  //     )
  //   )
  // );

  return {
    D1,
    D2
  };
};
var theta1 = Matrix.random(hidden, inputs + 1);
var theta2 = Matrix.random(outputs, hidden + 1);
const train = (X, y, alpha = 1) => {
  let {D1, D2} = costFunction(X, y, theta1, theta2, inputs, outputs, hidden);
  theta1 = Matrix.subtract(
    theta1, 
    Matrix.scale(
      Matrix.fromArray(D1.toArray().slice(1)), alpha
    )
  );
  theta2.subtract(Matrix.scale(D2, alpha));

  return forwardPropagate.bind(null, theta1, theta2, inputs);
};  

window.train = train;

},{}]},{},[1]);
