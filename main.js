#!/usr/bin/env node
const math = require('mathjs');

let hiddenNeurons = 3;
let inputs = 4;
let outputs = 1;
let theta1 = math.random([hiddenNeurons + 1, inputs + 1], 0, 1);
let theta2 = math.random([outputs, hiddenNeurons + 1], 0, 1);
let trainingCount = 1000;

const alpha = 10;

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

const forwardPropagate = (X, theta1, theta2) => {
  let a1 = math.subset(X, math.index(math.range(0, inputs + 1), 0));
  let a2 = matrixSigmoid(math.multiply(theta1, a1));
  let a3 = sigmoid(math.squeeze(math.multiply(theta2, a2)));

  return {
    a1,
    a2,
    a3 
  };
};

const costFunction = (theta1, theta2) => {
  let X = [
    [0, 0, 0, 1, 0], 
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0]
  ];
  let y = [
    [0],
    [0],
    [0],
    [1],
    [1]
  ];
  let m = math.size(X)[1];
  let n = math.size(X)[0];
  let bigDelta1 = math.zeros(hiddenNeurons + 1, inputs + 1);
  let bigDelta2 = math.zeros(outputs, hiddenNeurons + 1);
  let a3s = [
    [0],
    [0],
    [0],
    [0]
  ];

  X = math.ones(1, m).toArray().concat(X);


  for (let i = 0; i < m; i++) {
    let {a1, a2, a3} = forwardPropagate(math.subset(X, math.index(math.range(0, n + 1), i)), theta1, theta2);
    a3s[i] = [a3];
    let d3 = math.subtract(a3, math.squeeze(y[i]));
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
  let J = math.multiply(
            -(1/m), 
            math.add(
              math.multiply(
                math.transpose(y), 
                math.log(a3s)
              ), 
              math.multiply(
                math.subtract(1, math.transpose(y)), 
                math.log(math.subtract(1, a3s))
              )
            )
  );

  return {
    D1,
    D2,
    J
  };
};

for (let i = 0; i < trainingCount; i++) {
  let {D1, D2, J} = costFunction(theta1, theta2);

  theta1 = math.subtract(theta1, math.multiply(alpha, D1)).toArray();
  theta2 = math.subtract(theta2, math.multiply(alpha, D2)).toArray();
}

console.log(forwardPropagate([[1], [0], [0], [1], [0]], theta1, theta2).a3);
