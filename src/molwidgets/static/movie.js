// ECFPMovie: the "Inside ECFP4" explainer animation. The page is self-contained, so it runs
// in an iframe (its own document and keyboard shortcuts) holding only the 16:9 video; the
// control bar lives here, in the host page, and drives the frame with postMessage.

const CSS = `
.em-root { font: 12px/1.4 system-ui, sans-serif; color: #8a93a6; }
.em-frame { width: 100%; aspect-ratio: 16 / 9; border: 0; display: block; border-radius: 12px; background: #070a12; }
.em-ctrl { display: flex; align-items: flex-start; gap: 12px; padding: 8px 2px 0; }
.em-ctrl button { font: inherit; font-size: 16px; line-height: 20px; width: 26px; padding: 0; border: 0; background: none; color: inherit; cursor: pointer; }
.em-bar { position: relative; flex: 1; height: 34px; cursor: pointer; touch-action: none; }
.em-track, .em-fill { position: absolute; top: 8px; height: 4px; border-radius: 2px; }
.em-track { left: 0; right: 0; background: rgba(138, 147, 166, .3); }
.em-fill { left: 0; background: #4dabf7; }
.em-tick { position: absolute; top: 4px; width: 2px; height: 12px; background: rgba(138, 147, 166, .6); }
.em-tick span { position: absolute; top: 14px; left: 3px; white-space: nowrap; font-size: 11px; }
.em-time { font-variant-numeric: tabular-nums; min-width: 84px; text-align: right; line-height: 20px; }
`;

const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "em-root";
  root.innerHTML = `<style>${CSS}</style>
    <iframe class="em-frame" title="Inside ECFP4 animation"></iframe>
    <div class="em-ctrl"><button title="play / pause">▶</button>
      <div class="em-bar"><div class="em-track"></div><div class="em-fill"></div></div>
      <span class="em-time">0:00</span></div>`;
  el.appendChild(root);
  const frame = root.querySelector("iframe"), btn = root.querySelector("button");
  const bar = root.querySelector(".em-bar"), fill = root.querySelector(".em-fill"), time = root.querySelector(".em-time");
  frame.srcdoc = model.get("page");

  let end = 0, ticks = false;
  const send = (msg) => frame.contentWindow?.postMessage({ type: "ecfp-movie-cmd", ...msg }, "*");
  const onMessage = (ev) => {
    const m = ev.data;
    if (ev.source !== frame.contentWindow || !m || m.type !== "ecfp-movie") return;
    end = m.end;
    btn.textContent = m.playing ? "❚❚" : "▶";
    fill.style.width = `${(m.t / m.end) * 100}%`;
    time.textContent = `${mmss(m.t)} / ${mmss(m.end)}`;
    if (!ticks) {
      ticks = true;
      for (const [ct, name] of m.chapters) {
        const d = document.createElement("div");
        d.className = "em-tick";
        d.style.left = `${(ct / m.end) * 100}%`;
        d.innerHTML = `<span>${name}</span>`;
        bar.appendChild(d);
      }
    }
  };
  window.addEventListener("message", onMessage);
  btn.addEventListener("click", () => send({ cmd: "toggle" }));
  const seek = (ev) => {
    const r = bar.getBoundingClientRect();
    if (end) send({ cmd: "seek", t: Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)) * end });
  };
  bar.addEventListener("pointerdown", (ev) => {
    bar.setPointerCapture(ev.pointerId);
    seek(ev);
    bar.onpointermove = seek;
    bar.onpointerup = () => (bar.onpointermove = null);
  });
  return () => window.removeEventListener("message", onMessage);
}

export default { render };
