/*
 * The trained PPO policy, running in the browser.
 *
 * A port of ActorCritic from ppo-mario/src/train.py: three convolutions, one
 * hidden layer, then an action head and a value head. Weights arrive as int8
 * with one scale per output channel and are dequantized to float32 on load, so
 * the arithmetic here matches PyTorch to within the quantization error.
 *
 * The preprocessing mirrors ppo-mario/src/env.py: OpenCV's integer grayscale,
 * an INTER_AREA resize from 256x240 to 84x84, and a stack of the last four
 * kept frames.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MarioPolicy = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var W = 84, H = 84, STACK = 4;
  var SRC_W = 256, SRC_H = 240;

  /*
   * jsnes and the nes-py emulator used for training approximate the NES colours
   * differently, so the same pixel arrives at a different RGB in each. The
   * network only ever saw nes-py's numbers, so every colour jsnes produces is
   * translated to nes-py's before going grey. Derived by pairing pixels of the
   * same frame in both emulators; level 1-1 only ever uses these twelve.
   */
  var PALETTE = [
    [129, 121, 255, 104, 136, 252],
    [177, 84, 0, 228, 92, 16],
    [0, 0, 0, 0, 0, 0],
    [255, 209, 199, 240, 208, 176],
    [0, 171, 0, 0, 168, 0],
    [255, 255, 255, 252, 252, 252],
    [117, 227, 0, 184, 248, 24],
    [91, 106, 0, 172, 124, 0],
    [188, 25, 0, 248, 56, 0],
    [247, 180, 0, 252, 160, 68],
    [60, 171, 255, 60, 188, 252],
    [64, 24, 0, 136, 20, 0]
  ];

  /* OpenCV's fixed-point RGB to grey, which is what cv2.cvtColor applied in training. */
  function grey(r, g, b) {
    return (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
  }

  /* Colour key: the top 5 bits of each channel, which separates all 64 NES colours. */
  function key(r, g, b) {
    return ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
  }

  /*
   * Grey for every colour, indexed by the top 5 bits of each channel. Known
   * colours get nes-py's grey; anything else (another level, a power-up state)
   * falls back to its own grey rather than to black.
   */
  var GREY = (function () {
    var lut = new Uint8Array(1 << 15), i;
    for (i = 0; i < lut.length; i++) {
      lut[i] = grey(((i >> 10) & 31) << 3, ((i >> 5) & 31) << 3, (i & 31) << 3);
    }
    for (i = 0; i < PALETTE.length; i++) {
      var p = PALETTE[i];
      lut[key(p[0], p[1], p[2])] = grey(p[3], p[4], p[5]);
    }
    return lut;
  })();

  /*
   * The same translation for what gets drawn, so the page looks like the GIFs in
   * the article rather than like jsnes. Canvas wants 0xAABBGGRR; unknown colours
   * are passed through, since jsnes already packs them that way.
   */
  var COLOUR = (function () {
    var lut = new Uint32Array(1 << 15); // 0 means "not one of nes-py's, pass it through"
    for (var i = 0; i < PALETTE.length; i++) {
      var p = PALETTE[i];
      lut[key(p[0], p[1], p[2])] = 0xff000000 | (p[5] << 16) | (p[4] << 8) | p[3];
    }
    return lut;
  })();

  function recolour(framebuffer, pixels) {
    for (var i = 0; i < framebuffer.length; i++) {
      var v = framebuffer[i];
      pixels[i] = COLOUR[((v & 0xf8) << 7) | ((v >> 6) & 0x3e0) | ((v >> 19) & 0x1f)] || (0xff000000 | v);
    }
  }

  /* The same translation for one colour, for pixels taken straight from a palette. */
  function colour(v) {
    return COLOUR[((v & 0xf8) << 7) | ((v >> 6) & 0x3e0) | ((v >> 19) & 0x1f)] || (0xff000000 | v);
  }

  /* ---- tensor ops (NCHW, no padding) ---- */

  function conv(x, xc, xs, w, b, oc, k, stride) {
    var os = ((xs - k) / stride + 1) | 0;
    var out = new Float32Array(oc * os * os);
    var kk = k * k, plane = xs * xs, oplane = os * os;
    for (var o = 0; o < oc; o++) {
      var wo = o * xc * kk, bias = b[o], oo = o * oplane;
      for (var y = 0; y < os; y++) {
        for (var xx = 0; xx < os; xx++) {
          var sum = bias, sy = y * stride, sx = xx * stride;
          for (var c = 0; c < xc; c++) {
            var xb = c * plane, wc = wo + c * kk;
            for (var ky = 0; ky < k; ky++) {
              var row = xb + (sy + ky) * xs + sx, wr = wc + ky * k;
              for (var kx = 0; kx < k; kx++) sum += x[row + kx] * w[wr + kx];
            }
          }
          out[oo + y * os + xx] = sum > 0 ? sum : 0;
        }
      }
    }
    return out;
  }

  function linear(x, w, b, outN, inN, relu) {
    var out = new Float32Array(outN);
    for (var o = 0; o < outN; o++) {
      var sum = b[o], base = o * inN;
      for (var i = 0; i < inN; i++) sum += x[i] * w[base + i];
      out[o] = relu && sum < 0 ? 0 : sum;
    }
    return out;
  }

  /* ---- INTER_AREA weight tables ---- */

  function areaTable(srcLen, dstLen) {
    var scale = srcLen / dstLen, table = [];
    for (var i = 0; i < dstLen; i++) {
      var a = i * scale, b = (i + 1) * scale;
      var from = Math.floor(a), to = Math.ceil(b), weights = [];
      for (var k = from; k < to; k++) {
        weights.push(Math.min(k + 1, b) - Math.max(k, a));
      }
      table.push({ from: from, weights: weights });
    }
    return table;
  }

  var TX = areaTable(SRC_W, W), TY = areaTable(SRC_H, H);
  var AREA = (SRC_W / W) * (SRC_H / H);

  /* ---- frame -> 84x84 grayscale ---- */

  function grayscale(framebuffer, gray) {
    for (var i = 0, n = SRC_W * SRC_H; i < n; i++) {
      // jsnes packs each pixel as 0x00BBGGRR
      var v = framebuffer[i];
      gray[i] = GREY[(((v & 0xf8) << 7) | ((v >> 6) & 0x3e0) | ((v >> 19) & 0x1f))];
    }
  }

  function resize(gray, out) {
    for (var oy = 0; oy < H; oy++) {
      var ty = TY[oy];
      for (var ox = 0; ox < W; ox++) {
        var tx = TX[ox], sum = 0;
        for (var j = 0; j < ty.weights.length; j++) {
          var row = (ty.from + j) * SRC_W, wy = ty.weights[j], acc = 0;
          for (var i = 0; i < tx.weights.length; i++) acc += gray[row + tx.from + i] * tx.weights[i];
          sum += acc * wy;
        }
        var v = Math.round(sum / AREA);
        out[oy * W + ox] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }

  /* ---- public ---- */

  function Policy(meta, buffer) {
    var bytes = new Int8Array(buffer), offset = 0;
    this.layers = {};
    for (var i = 0; i < meta.layers.length; i++) {
      var L = meta.layers[i];
      var count = L.shape.reduce(function (a, b) { return a * b; }, 1);
      var inPer = count / L.shape[0];
      var w = new Float32Array(count);
      for (var o = 0; o < L.shape[0]; o++) {
        var s = L.scales[o], base = o * inPer;
        for (var j = 0; j < inPer; j++) w[base + j] = bytes[offset + base + j] * s;
      }
      offset += count;
      this.layers[L.name] = { w: w, b: Float32Array.from(L.bias), shape: L.shape };
    }
  }

  Policy.prototype.forward = function (state) {
    var x = new Float32Array(STACK * W * H);
    for (var i = 0; i < x.length; i++) x[i] = state[i] / 255;
    var l = this.layers;
    var h = conv(x, 4, 84, l.conv1.w, l.conv1.b, 32, 8, 4);
    h = conv(h, 32, 20, l.conv2.w, l.conv2.b, 64, 4, 2);
    h = conv(h, 64, 9, l.conv3.w, l.conv3.b, 64, 3, 1);
    var f = linear(h, l.fc1.w, l.fc1.b, 512, 3136, true);
    return {
      logits: linear(f, l.pi.w, l.pi.b, l.pi.shape[0], 512, false),
      value: linear(f, l.v.w, l.v.b, 1, 512, false)[0]
    };
  };

  Policy.prototype.act = function (state, sample) {
    var out = this.forward(state), logits = out.logits, i;
    var best = 0;
    for (i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    if (!sample) return best;
    var max = logits[best], total = 0, probs = new Float64Array(logits.length);
    for (i = 0; i < logits.length; i++) { probs[i] = Math.exp(logits[i] - max); total += probs[i]; }
    var r = Math.random() * total;
    for (i = 0; i < logits.length; i++) { r -= probs[i]; if (r <= 0) return i; }
    return best;
  };

  function Frames() {
    this.gray = new Int32Array(SRC_W * SRC_H);
    this.frame = new Uint8Array(W * H);
    this.state = new Uint8Array(STACK * W * H);
  }

  Frames.prototype.push = function (framebuffer, fill) {
    grayscale(framebuffer, this.gray);
    resize(this.gray, this.frame);
    if (fill) {
      for (var s = 0; s < STACK; s++) this.state.set(this.frame, s * W * H);
      return this.state;
    }
    this.state.copyWithin(0, W * H);
    this.state.set(this.frame, (STACK - 1) * W * H);
    return this.state;
  };

  return {
    Policy: Policy,
    Frames: Frames,
    recolour: recolour,
    colour: colour,
    WIDTH: W,
    HEIGHT: H,
    STACK: STACK
  };
});
