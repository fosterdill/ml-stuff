#!/usr/bin/env node
const math = require('mathjs');
const sigmoid = value => {
  return (1 / (1 + math.exp(-value)));
};

const matrixSigmoid = matrix => {
  return matrix.map(value => {
    return value.map(value => {
      return sigmoid(value);
    });
  });
};

const forwardPropagate = (theta1, theta2, inputs, x) => {
  let a1 = x;
  let a2 = matrixSigmoid(math.multiply(theta1, a1));
  a2 = math.ones(1, math.size(a2)[1]).toArray().concat(a2);
  let a3 = matrixSigmoid(math.multiply(theta2, a2));

  return {
    a1,
    a2,
    a3 
  };
};

const costFunction = (X, y, theta1, theta2, inputs, outputs, hidden) => {
  let m = math.size(X)[1];
  let n = math.size(X)[0];
  let bigDelta1 = math.zeros(hidden + 1, inputs + 1);
  let bigDelta2 = math.zeros(outputs, hidden + 1);
  let a3s = [];
  X = math.ones(1, m).toArray().concat(X);

  for (let i = 0; i < m; i++) {
    let {
      a1, 
      a2, 
      a3
    } = forwardPropagate(
      theta1, 
      theta2, 
      inputs,
      math.subset(X, math.index(math.range(0, inputs + 1), i))
    );

    a3s[i] = math.squeeze(a3);
    yi = math.subset(y, math.index(math.range(0, outputs), i))
    let d3 = math.subtract(a3, yi);
    let d2 = math.dotMultiply(
      math.dotMultiply(
        math.multiply(
          math.transpose(theta2), 
          d3
        ), 
        a2
      ), 
      math.subtract(1, a2)
    );

    bigDelta1 = math.add(bigDelta1, math.multiply(d2, math.transpose(a1)));
    bigDelta2 = math.add(bigDelta2, math.multiply(d3, math.transpose(a2)));
  }

  let D1 = math.multiply(1/m, bigDelta1);
  let D2 = math.multiply(1/m, bigDelta2);

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
let hidden = 15;
let inputs = 2;
let outputs = 3;
var theta1 = math.random([hidden, inputs + 1], -1, 1);
var theta2 = math.random([outputs, hidden + 1], -1, 1);

const train = (X, y, alpha = 1) => {
  let {D1, D2} = costFunction(X, y, theta1, theta2, inputs, outputs, hidden);

  theta1 = math.subtract(theta1, math.multiply(alpha, D1.toArray().slice(1)));
  theta2 = math.subtract(theta2, math.multiply(alpha, D2.toArray()));

  return forwardPropagate.bind(null, theta1, theta2, inputs);
};

window.train = train;
