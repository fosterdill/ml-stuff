window = self;
importScripts("node_modules/vectorious/dist/vectorious.min.js", "bundle.js");
var trainingSet = null;
var targetSet = null;
var alpha = null;
var input = Matrix.zeros(3, 1);
var colors = new Array(80 * 80 * 4);
input.set(0, 0, 1);
onmessage = (e) => {
    if (e.data.message === 'init') {
        trainingSet = Matrix.fromArray(e.data.trainingSet);
        targetSet = Matrix.fromArray(e.data.targetSet);
        alpha = e.data.alpha;
    } else if (e.data.message === 'train') {
        var h = self.train(trainingSet, targetSet, alpha);
        for (var i = 0; i < (80 * 80 * 4); i += 4) {
            input.set(1, 0, ((i / 4) % 80) / 80);
            input.set(2, 0, Math.floor((i / 4) / 80) / 80);

            var color = h(input).a3;

            colors[i + 0] = color.get(0, 0) * 255;
            colors[i + 1] = color.get(1, 0) * 255;
            colors[i + 2] = color.get(2, 0) * 255;
            colors[i + 3] = 255;
        }

        postMessage(colors);
    }
};