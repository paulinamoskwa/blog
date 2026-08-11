/*
 * A ghost race against the PPO agent, in the real Super Mario Bros.
 *
 * You play the visible emulator. A second one runs out of sight with the agent
 * from ppo-mario driving it, live: every fourth frame its framebuffer is
 * reduced to the four stacked 84x84 grayscale images the network was trained
 * on, the network samples one of the seven SIMPLE_MOVEMENT actions, and that
 * action is held for the next four frames. Wherever it has got to is drawn over
 * your screen as a translucent Mario, so the ghost is where the agent actually
 * is at this moment, not a recording.
 *
 * Before you press play the agent's own screen is shown instead, so the level
 * is being played the whole time the page is open.
 *
 * Both machines start together. Dying, finishing or pressing R starts a new
 * race for both. If the agent dies on its own it quietly starts over.
 *
 * The boot sequence and the RAM addresses mirror
 * gym_super_mario_bros/smb_env.py so the agent meets the states it trained on.
 */
(function () {
  'use strict';

  var SKIP = 4;
  var FLAG_X = 3161;
  var SCREEN_W = 256, SCREEN_H = 240;

  // RAM, per smb_env.py and the Super Mario Bros RAM map
  var RAM_PLAYER_STATE = 0x000e;
  var RAM_Y_VIEWPORT = 0x00b5;
  var RAM_X_PAGE = 0x006d;
  var RAM_X = 0x0086;
  var RAM_Y_PIXEL = 0x03b8;
  var RAM_CAMERA_PAGE = 0x071a;
  var RAM_CAMERA_X = 0x071c;
  var RAM_LIFE = 0x075a;
  var RAM_PRELEVEL_TIMER = 0x07a0;
  var RAM_CHANGE_AREA = 0x06de;
  var RAM_TIME = 0x07f8;
  var BUSY_STATES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07];

  // SIMPLE_MOVEMENT, as button names
  var ACTIONS = [[], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']];

  /*
   * Mario is drawn as OAM entries 1 to 8: two columns of 8x8 tiles, four rows
   * deep, the first row sixteen pixels above the value at RAM_Y_PIXEL. Small
   * Mario leaves the top two rows as a blank tile. Sprite 0 is the coin in the
   * status bar, which the game uses for its sprite-zero split, so the range
   * below is the player and nothing else: an enemy standing on top of him
   * cannot end up in the ghost.
   */
  var OAM_FIRST = 1, OAM_LAST = 8, OAM_TOP = 1;
  var GHOST_W = 18, GHOST_H = 34, GHOST_PAD = 1;
  var GHOST_ALPHA = 0.62;
  var HALO = 0xff1a1a1a;
  var COUNT_MS = 620;

  function Machine(rom) {
    this.rom = rom;
    this.framebuffer = null;
    this.image = null;
    var C = window.jsnes.Controller;
    this.B = {
      right: C.BUTTON_RIGHT, left: C.BUTTON_LEFT, up: C.BUTTON_UP, down: C.BUTTON_DOWN,
      A: C.BUTTON_A, B: C.BUTTON_B, start: C.BUTTON_START, select: C.BUTTON_SELECT
    };
    this.held = [];
    this.boot();
  }

  Machine.prototype.mem = function () { return this.nes.cpu.mem; };

  Machine.prototype.frame = function (names) {
    names = names || [];
    var i;
    for (i = 0; i < this.held.length; i++) this.nes.buttonUp(1, this.B[this.held[i]]);
    for (i = 0; i < names.length; i++) this.nes.buttonDown(1, this.B[names[i]]);
    this.held = names.slice();
    this.nes.frame();
  };

  Machine.prototype.paint = function (ctx) {
    if (!this.framebuffer) return;
    if (!this.image) {
      this.image = ctx.createImageData(SCREEN_W, SCREEN_H);
      this.pixels = new Uint32Array(this.image.data.buffer);
    }
    // repainted in nes-py's colours so the page matches the GIFs in the article
    window.MarioPolicy.recolour(this.framebuffer, this.pixels);
    ctx.putImageData(this.image, 0, 0);
  };

  Machine.prototype.time = function () {
    var m = this.mem();
    return m[RAM_TIME] * 100 + m[RAM_TIME + 1] * 10 + m[RAM_TIME + 2];
  };
  Machine.prototype.x = function () {
    var m = this.mem();
    return m[RAM_X_PAGE] * 256 + m[RAM_X];
  };
  Machine.prototype.camera = function () {
    var m = this.mem();
    return m[RAM_CAMERA_PAGE] * 256 + m[RAM_CAMERA_X];
  };
  Machine.prototype.y = function () { return this.mem()[RAM_Y_PIXEL]; };
  Machine.prototype.state = function () { return this.mem()[RAM_PLAYER_STATE]; };
  Machine.prototype.busy = function () { return BUSY_STATES.indexOf(this.state()) !== -1; };
  Machine.prototype.lost = function () {
    var s = this.state();
    return s === 0x0b || s === 0x06 || this.mem()[RAM_Y_VIEWPORT] > 1 || this.mem()[RAM_LIFE] === 0xff;
  };
  Machine.prototype.won = function () { return this.x() >= FLAG_X; };

  /*
   * Reproduces _skip_start_screen from smb_env.py. Three details have to match
   * the training emulator, or this ends up in the attract mode demo that plays
   * behind the title screen rather than in the level:
   *   - a fresh NES each time, since loadROM leaves the previous run's PPU
   *     memory in place and the agent then behaves differently run to run;
   *   - zeroed RAM at power-on, which is what nes-py does; jsnes fills it with
   *     0xff, which makes the timer read non-zero and skips the loop below;
   *   - the full 256x240 frame, since jsnes blanks eight pixels on each edge.
   */
  Machine.prototype.boot = function () {
    var self = this;
    this.nes = new window.jsnes.NES({
      onFrame: function (fb) { self.framebuffer = fb; },
      onAudioSample: null
    });
    this.nes.loadROM(this.rom);
    this.nes.ppu.clipToTvSize = false; // after loadROM: reset() inside it replaces the PPU object
    var mem = this.mem(), i;
    for (i = 0; i < 0x800; i++) mem[i] = 0;
    this.held = [];

    var guard = 0;
    this.frame(['start']);
    this.frame([]);
    while (this.time() === 0 && guard++ < 2000) {
      this.frame(['start']);
      this.frame([]);
      this.mem()[RAM_PRELEVEL_TIMER] = 0;
    }
    var last = this.time();
    guard = 0;
    while (this.time() >= last && guard++ < 2000) {
      last = this.time();
      this.frame(['start']);
      this.frame([]);
    }
    guard = 0;
    while (this.busy() && guard++ < 2000) {
      this.mem()[RAM_PRELEVEL_TIMER] = 0;
      this.frame([]);
    }
  };

  Machine.prototype.housekeep = function () {
    var m = this.mem(), t = m[RAM_CHANGE_AREA];
    if (t > 1 && t < 255) m[RAM_CHANGE_AREA] = 1;
  };

  /*
   * The ghost is Mario's own sprite, lifted out of the agent's OAM every frame
   * and redrawn, so it runs, turns and jumps exactly as the agent does rather
   * than sliding along in one fixed pose. A dark halo is grown around the
   * silhouette, without which a translucent Mario disappears against the sky,
   * which is where he spends every jump.
   */
  function Ghost() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = GHOST_W;
    this.canvas.height = GHOST_H;
    this.ctx = this.canvas.getContext('2d');
    this.image = this.ctx.createImageData(GHOST_W, GHOST_H);
    this.pixels = new Uint32Array(this.image.data.buffer);
    this.solid = new Uint8Array(GHOST_W * GHOST_H);
  }

  Ghost.prototype.capture = function (machine) {
    var ppu = machine.nes.ppu, oam = ppu.spriteMem;
    var mx = machine.x() - machine.camera(), my = machine.y();
    var found = false, s, o, sy, tile, attr, sx, t, palAdd, flipH, flipV, bx, by, r, c, px, py, v;

    this.pixels.fill(0);
    this.solid.fill(0);

    for (s = OAM_FIRST; s <= OAM_LAST; s++) {
      o = s * 4;
      sy = oam[o] + 1;
      tile = oam[o + 1];
      attr = oam[o + 2];
      sx = oam[o + 3];
      if (sy >= SCREEN_H) continue;
      t = ppu.ptTile[ppu.f_spPatternTable === 0 ? tile : tile + 256];
      if (!t || !t.initialized) continue;
      palAdd = (attr & 3) << 2;
      flipH = (attr >> 6) & 1;
      flipV = (attr >> 7) & 1;
      bx = sx - mx + GHOST_PAD;
      by = sy - my - OAM_TOP + GHOST_PAD;
      for (r = 0; r < 8; r++) {
        py = by + r;
        if (py < 0 || py >= GHOST_H) continue;
        for (c = 0; c < 8; c++) {
          px = bx + c;
          if (px < 0 || px >= GHOST_W) continue;
          v = t.pix[((flipV ? 7 - r : r) << 3) + (flipH ? 7 - c : c)];
          if (!v) continue; // colour 0 is transparent for sprites
          this.pixels[py * GHOST_W + px] = window.MarioPolicy.colour(ppu.sprPalette[palAdd + v]);
          this.solid[py * GHOST_W + px] = 1;
          found = true;
        }
      }
    }
    if (!found) return false;

    var solid = this.solid, pix = this.pixels, i;
    for (py = 0; py < GHOST_H; py++) {
      for (px = 0; px < GHOST_W; px++) {
        i = py * GHOST_W + px;
        if (solid[i]) continue;
        if ((px > 0 && solid[i - 1]) || (px < GHOST_W - 1 && solid[i + 1]) ||
            (py > 0 && solid[i - GHOST_W]) || (py < GHOST_H - 1 && solid[i + GHOST_W])) {
          pix[i] = HALO;
        }
      }
    }
    this.ctx.putImageData(this.image, 0, 0);
    return true;
  };

  function init(root) {
    var dom = document.createElement('div');
    dom.className = 'duel';
    dom.innerHTML =
      '<div class="duel-screen">' +
        '<canvas width="256" height="240"></canvas>' +
        '<div class="duel-count" aria-hidden="true"></div>' +
        '<div class="duel-overlay is-loading">' +
          '<span class="duel-status">loading the emulator and the weights</span>' +
          '<div class="duel-controls">' +
            '<div class="duel-buttons">' +
              '<button type="button" class="duel-btn duel-play" aria-label="Play">' +
                '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.2v13.6L19 12z"/></svg>' +
              '</button>' +
              '<button type="button" class="duel-btn duel-stop" aria-label="Start over">' +
                '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>' +
              '</button>' +
            '</div>' +
            '<dl class="duel-keys">' +
              '<dt><kbd>&#8592;</kbd><kbd>&#8594;</kbd></dt><dd>move</dd>' +
              '<dt><kbd>space</kbd></dt><dd><b>A</b>, jump</dd>' +
              '<dt><kbd>Z</kbd></dt><dd><b>B</b>, run</dd>' +
              '<dt><kbd>R</kbd></dt><dd>start the race over</dd>' +
            '</dl>' +
            '<p class="duel-note">click the screen to pause</p>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div class="duel-meta"><span data-meta-you>&nbsp;</span><span data-meta-agent></span></div>';

    var pad = document.createElement('div');
    pad.className = 'duel-pad';
    pad.innerHTML =
      '<button type="button" data-key="left" aria-label="left">&#9664;</button>' +
      '<button type="button" data-key="right" aria-label="right">&#9654;</button>' +
      '<button type="button" data-key="B" aria-label="B, run">B<small>run</small></button>' +
      '<button type="button" data-key="A" aria-label="A, jump">A<small>jump</small></button>';

    root.appendChild(dom);
    root.appendChild(pad);

    var screen = dom.querySelector('.duel-screen');
    var ctx = dom.querySelector('canvas').getContext('2d');
    var count = dom.querySelector('.duel-count');
    var overlay = dom.querySelector('.duel-overlay');
    var status = dom.querySelector('.duel-status');
    var playBtn = dom.querySelector('.duel-play');
    var stopBtn = dom.querySelector('.duel-stop');
    var metaYou = dom.querySelector('[data-meta-you]');
    var metaAgent = dom.querySelector('[data-meta-agent]');

    var you = null, agent = null, policy = null, frames = null, ghost = null;
    var mode = 'loading'; // loading | idle | countdown | playing | paused | error
    var keys = {}, visible = true, fresh = true, raf = 0;
    var tick = 0, action = 0;
    var waitYou = 0, waitAgent = 0;
    var countEnd = 0, countShown = null;
    var FRAME_MS = 1000 / 60, lastTs = 0, accumulator = 0;

    var KEYMAP = {
      ArrowLeft: 'left', KeyA: 'left', ArrowRight: 'right', KeyD: 'right',
      ArrowDown: 'down', KeyS: 'down', ArrowUp: 'A', KeyW: 'A',
      Space: 'A', KeyX: 'A', KeyZ: 'B'
    };
    // the page's own scrolling keys, which are left alone until the race is on
    var SCROLLS = { ArrowLeft: 1, ArrowRight: 1, ArrowUp: 1, ArrowDown: 1, Space: 1 };

    function pressed() {
      var out = [];
      for (var k in keys) if (keys[k]) out.push(k);
      return out;
    }

    function setMode(next) {
      mode = next;
      overlay.classList.toggle('is-loading', next === 'loading' || next === 'error');
      overlay.classList.toggle('is-hidden', next === 'playing' || next === 'countdown');
      playBtn.setAttribute('aria-label', next === 'paused' ? 'Resume' : 'Play');
      if (next === 'idle' || next === 'paused') keys = {};
      if ((next === 'playing' || next === 'countdown') && !raf) {
        lastTs = 0;
        accumulator = 0;
        raf = requestAnimationFrame(loop);
      }
    }

    /* Neither emulator advances until the first press of play. */
    function idle() {
      setMode('idle');
      showCount(0);
      you.paint(ctx);
      metaYou.innerHTML = '&nbsp;';
      metaAgent.textContent = '';
    }

    function showCount(n) {
      if (n === countShown) return;
      countShown = n;
      count.textContent = n || '';
      count.classList.remove('is-on');
      if (n) {
        void count.offsetWidth; // reflow, so the pop animation runs again for each number
        count.classList.add('is-on');
      }
    }

    /* Three, two, one before every start, so no race begins under your feet. */
    function startCountdown() {
      countEnd = performance.now() + 3 * COUNT_MS;
      you.paint(ctx);
      setMode('countdown');
    }

    function binary(url) {
      return fetch(url).then(function (r) {
        if (!r.ok) throw new Error(url.split('/').pop() + ' ' + r.status);
        return r.arrayBuffer();
      });
    }

    function load() {
      Promise.all([
        binary(root.getAttribute('data-rom')),
        fetch(root.getAttribute('data-policy-meta')).then(function (r) { return r.json(); }),
        binary(root.getAttribute('data-policy'))
      ]).then(function (res) {
        var bytes = new Uint8Array(res[0]), rom = '', i;
        for (i = 0; i < bytes.length; i++) rom += String.fromCharCode(bytes[i]);
        policy = new window.MarioPolicy.Policy(res[1], res[2]);
        frames = new window.MarioPolicy.Frames();
        ghost = new Ghost();
        you = new Machine(rom);
        agent = new Machine(rom);
        frames.push(agent.framebuffer, true);
        idle();
      }).catch(function (e) {
        status.textContent = 'could not load: ' + e.message;
        setMode('error');
      });
    }

    function restartAgent() {
      agent.boot();
      frames.push(agent.framebuffer, true);
      tick = 0;
      action = 0;
      waitAgent = 0;
    }

    /* A new race: both machines back to the start together. */
    function restartRace() {
      you.boot();
      restartAgent();
      waitYou = 0;
      fresh = true;
    }

    function play() {
      if (mode !== 'idle' && mode !== 'paused') return;
      if (mode === 'idle' && !fresh) restartRace(); // already at the start on first press
      startCountdown();
      playBtn.blur();
    }

    function stop() {
      if (mode === 'loading' || mode === 'error') return;
      if (!fresh) restartRace();
      idle();
      stopBtn.blur();
    }

    function stepYou() {
      if (waitYou > 0) {
        if (--waitYou === 0) { restartRace(); startCountdown(); }
        return;
      }
      you.housekeep();
      if (you.lost()) {
        waitYou = 90;
        metaYou.textContent = 'you died at ' + you.x();
      } else if (you.won()) {
        waitYou = 150;
        metaYou.textContent = 'you got the flag';
      } else {
        you.frame(pressed());
        metaYou.textContent = 'you ' + you.x();
      }
    }

    function stepAgent() {
      if (waitAgent > 0) {
        if (--waitAgent === 0) restartAgent();
        return;
      }
      agent.housekeep();
      if (agent.lost()) {
        waitAgent = 60; // quietly start over, as if it had never been there
      } else if (agent.won()) {
        waitAgent = 150;
        metaAgent.textContent = 'agent got the flag';
      } else {
        agent.frame(ACTIONS[action]);
        if (++tick % SKIP === 0) action = policy.act(frames.push(agent.framebuffer, false), true);
      }
    }

    /* The agent's Mario, drawn where it actually is, in your screen's coordinates. */
    function drawGhost() {
      if (waitAgent > 0 || waitYou > 0) return;
      var sx = agent.x() - you.camera();
      var sy = agent.y() + OAM_TOP;

      ctx.save();
      ctx.globalAlpha = GHOST_ALPHA;
      if (sx > SCREEN_W - 12) {
        // ahead and off the right edge: a marker on the edge instead
        ctx.fillStyle = '#f83800';
        ctx.beginPath();
        ctx.moveTo(SCREEN_W - 3, Math.max(10, Math.min(SCREEN_H - 10, sy + 12)));
        ctx.lineTo(SCREEN_W - 11, Math.max(4, Math.min(SCREEN_H - 16, sy + 6)));
        ctx.lineTo(SCREEN_W - 11, Math.min(SCREEN_H - 4, sy + 18));
        ctx.fill();
      } else if (sx > -18 && ghost.capture(agent)) {
        ctx.drawImage(ghost.canvas, Math.round(sx) - GHOST_PAD, Math.round(sy) - GHOST_PAD);
      }
      ctx.restore();

      var gap = agent.x() - you.x();
      metaAgent.textContent = gap === 0 ? 'level with you' :
        gap > 0 ? 'agent ' + gap + ' ahead' : 'agent ' + (-gap) + ' behind';
    }

    /*
     * The NES runs at 60 Hz whatever the display does, so steps are taken from
     * elapsed time rather than once per frame, or the game runs at double speed
     * on a 120 Hz screen. Catch-up is capped so a backgrounded tab does not
     * come back and try to emulate thousands of frames at once.
     */
    function loop(ts) {
      raf = 0;
      if (mode !== 'playing' && mode !== 'countdown') return; // pausing lets the pump run down
      raf = requestAnimationFrame(loop);

      if (mode === 'countdown') {
        var left = countEnd - ts;
        if (left > 0) { showCount(Math.ceil(left / COUNT_MS)); return; }
        showCount(0);
        setMode('playing');
        lastTs = 0;
        accumulator = 0;
      }

      if (!visible) { lastTs = ts; return; }
      accumulator = Math.min(accumulator + (lastTs ? ts - lastTs : FRAME_MS), 4 * FRAME_MS);
      lastTs = ts;
      var steps = 0;
      while (accumulator >= FRAME_MS && steps++ < 4 && mode === 'playing') {
        accumulator -= FRAME_MS;
        fresh = false;
        stepYou();
        stepAgent();
      }
      you.paint(ctx);
      drawGhost();
    }

    window.addEventListener('keydown', function (e) {
      if (mode === 'loading' || mode === 'error') return;
      var k = KEYMAP[e.code];
      if (e.code === 'KeyR') {
        if (mode === 'playing') restartRace();
        return;
      }
      if (!k) return;
      if (mode === 'idle' || mode === 'paused') {
        if (SCROLLS[e.code]) return; // reading the page, not playing yet
        e.preventDefault();
        play(); // reaching for the buttons means go
        return;
      }
      e.preventDefault();
      keys[k] = true; // holding a key through the count still counts
    });
    window.addEventListener('keyup', function (e) {
      var k = KEYMAP[e.code];
      if (k) keys[k] = false;
    });

    screen.addEventListener('click', function () { if (mode === 'playing') setMode('paused'); });
    // stopped here, or the same click would bubble on and pause what it just started
    overlay.addEventListener('click', function (e) { e.stopPropagation(); play(); });
    stopBtn.addEventListener('click', function (e) { e.stopPropagation(); stop(); });

    Array.prototype.forEach.call(pad.querySelectorAll('button'), function (b) {
      var k = b.getAttribute('data-key');
      var on = function (e) { e.preventDefault(); if (mode === 'playing') keys[k] = true; };
      var off = function (e) { e.preventDefault(); keys[k] = false; };
      b.addEventListener('touchstart', on, { passive: false });
      b.addEventListener('touchend', off);
      b.addEventListener('touchcancel', off);
      b.addEventListener('mousedown', on);
      b.addEventListener('mouseup', off);
      b.addEventListener('mouseleave', off);
    });

    if (window.IntersectionObserver) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
      }, { threshold: 0.05 }).observe(dom);
    }

    load();
  }

  function boot() {
    Array.prototype.forEach.call(document.querySelectorAll('.race[data-rom]'), init);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
