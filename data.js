class Dataset {
    constructor() {
        this.labels = []
    }
    
    addExample(example, label) {
        if (this.xs == null) {
            this.xs = tf.keep(example);
            this.labels.push(label);
        } else {
            const Oldx = this.xs;
            this.xs = tf.keep(Oldx.concat(example, 0));
            this.labels.push(label);
            Oldx.dispose();
        }
    }
    
    encodeLabels(numClasses) {
        for (var i = 0; i< this.labels.length; i++) {
            if (this.ys == null) {
                this.ys = tf.keep(tf.tidy(() => {return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
            } else {
                const y = tf.tidy(() => {return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
                const oldy = this.ys;
                this.ys = tf.keep(oldy.concat(y, 0));
                oldy.dispose();
                y.dispose();
            }
        }
    }
}