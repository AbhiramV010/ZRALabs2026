# The token system and stylesheet for the report page.
#
# Light is the complete palette on bare :root. Dark redefines only tokens,
# once under the media query (guarded so an explicit light stamp wins) and
# once under the [data-theme="dark"] stamp. Nothing below styles a component
# from inside a media or [data-theme] block.

CSS = """
:root {
  color-scheme: light;
  --ground:  #f2f4f0;
  --surface: #fbfcfa;
  --sunk:    #eceee8;
  --ink:     #141a17;
  --ink-2:   #4b5551;
  --ink-3:   #7a847f;
  --rule:    #dde1da;
  --axis:    #c2c9bf;
  --accent:  #8a4b1f;
  --accent-soft: rgba(138,75,31,0.10);
  --s1: #2a78d6;
  --s2: #eb6834;
  --s3: #1baf7a;
  --good: #0ca30c;
  --warn: #b0741a;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground:  #101312;
    --surface: #181c1a;
    --sunk:    #1f2422;
    --ink:     #eef2ee;
    --ink-2:   #a9b3ae;
    --ink-3:   #7f8a85;
    --rule:    #2a302d;
    --axis:    #3a423e;
    --accent:  #cf8a55;
    --accent-soft: rgba(207,138,85,0.14);
    --s1: #3987e5;
    --s2: #d95926;
    --s3: #199e70;
    --good: #0ca30c;
    --warn: #d9a54a;
  }
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --ground:  #101312;
  --surface: #181c1a;
  --sunk:    #1f2422;
  --ink:     #eef2ee;
  --ink-2:   #a9b3ae;
  --ink-3:   #7f8a85;
  --rule:    #2a302d;
  --axis:    #3a423e;
  --accent:  #cf8a55;
  --accent-soft: rgba(207,138,85,0.14);
  --s1: #3987e5;
  --s2: #d95926;
  --s3: #199e70;
  --good: #0ca30c;
  --warn: #d9a54a;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}

.mono, .tick, .bar-value, .row-label, .panel-head, .note-in, .end-label,
.eyebrow, table, .kpi-value, .kpi-unit, .machine, .stamp {
  font-family: ui-monospace, "Cascadia Mono", "Cascadia Code", Consolas,
               "SF Mono", Menlo, monospace;
  font-variant-ligatures: none;
}

.wrap {
  max-width: 1080px;
  margin: 0 auto;
  padding: 0 28px 96px;
}

.masthead {
  border-bottom: 1px solid var(--rule);
  padding: 56px 0 26px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.stamp {
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
}

h1 {
  margin: 0;
  font-size: clamp(30px, 4.6vw, 46px);
  line-height: 1.08;
  letter-spacing: -0.025em;
  font-weight: 680;
  text-wrap: balance;
  max-width: 20ch;
}

.standfirst {
  margin: 0;
  max-width: 66ch;
  font-size: 17.5px;
  color: var(--ink-2);
}

.machine {
  font-size: 12px;
  color: var(--ink-3);
  display: flex;
  flex-wrap: wrap;
  gap: 6px 20px;
  padding-top: 4px;
}
.machine b { color: var(--ink-2); font-weight: 600; }

section { padding-top: 60px; }

.eyebrow {
  font-size: 11px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--ink-3);
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}
.eyebrow::after {
  content: "";
  flex: 1;
  height: 1px;
  background: var(--rule);
}
.eyebrow em {
  font-style: normal;
  color: var(--accent);
}

h2 {
  margin: 0 0 12px;
  font-size: clamp(22px, 2.6vw, 27px);
  letter-spacing: -0.018em;
  font-weight: 660;
  line-height: 1.18;
  text-wrap: balance;
}

h3 {
  margin: 34px 0 10px;
  font-size: 16.5px;
  letter-spacing: -0.008em;
  font-weight: 640;
}

p { margin: 0 0 14px; max-width: 68ch; }
p.lede { font-size: 17px; color: var(--ink-2); }

a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible, summary:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 3px;
  border-radius: 3px;
}

ul { max-width: 68ch; padding-left: 20px; margin: 0 0 14px; }
li { margin-bottom: 8px; }

strong { font-weight: 640; }
code {
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: 0.9em;
  background: var(--sunk);
  padding: 1px 5px;
  border-radius: 3px;
}

.figure {
  margin: 22px 0 8px;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 20px 20px 16px;
}

.figure-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 4px;
}
.figure-title { font-size: 15px; font-weight: 640; }
.figure-note { font-size: 12.5px; color: var(--ink-3); }

.chart { width: 100%; height: auto; display: block; }

.tick { font-size: 11.5px; fill: var(--ink-3); }
.axis-title { font-size: 11.5px; fill: var(--ink-3); letter-spacing: 0.06em; }
.bar-value { font-size: 11.5px; fill: var(--ink-2); }
.row-label { font-size: 12.5px; fill: var(--ink-2); }
.panel-head {
  font-size: 11px;
  fill: var(--ink-3);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.note-in { font-size: 11px; fill: var(--ink-3); }
.end-label { font-size: 12px; fill: var(--ink-2); }
.muted-label { fill: var(--ink-3); }

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
  margin-top: 14px;
  font-size: 12.5px;
  color: var(--ink-2);
}
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.swatch {
  width: 11px; height: 11px; border-radius: 2px; flex: none;
}

.scroll { overflow-x: auto; margin: 14px 0; }

table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}
caption {
  text-align: left;
  color: var(--ink-3);
  font-size: 12.5px;
  padding-bottom: 8px;
}
th, td {
  padding: 7px 12px;
  border-bottom: 1px solid var(--rule);
  white-space: nowrap;
}
thead th {
  color: var(--ink-3);
  font-weight: 600;
  font-size: 11px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  border-bottom-color: var(--axis);
}
tbody th { font-weight: 600; color: var(--ink); }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: none; }
.a-right { text-align: right; }
.a-left { text-align: left; }

.twin { margin: 10px 0 0; }
.twin summary {
  cursor: pointer;
  font-size: 12.5px;
  color: var(--ink-3);
  padding: 6px 0;
  list-style: none;
}
.twin summary::-webkit-details-marker { display: none; }
.twin summary::before { content: "+ "; color: var(--accent); }
.twin[open] summary::before { content: "- "; }
.twin summary:hover { color: var(--ink-2); }

.kpis {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 1px;
  background: var(--rule);
  border: 1px solid var(--rule);
  border-radius: 4px;
  overflow: hidden;
  margin: 22px 0;
}
.kpi { background: var(--surface); padding: 16px 18px; }
.kpi-label {
  font-size: 12px;
  color: var(--ink-3);
  margin-bottom: 6px;
}
.kpi-value {
  font-size: 26px;
  font-weight: 640;
  letter-spacing: -0.02em;
  line-height: 1.1;
  color: var(--ink);
}
.kpi-unit { font-size: 13px; color: var(--ink-3); font-weight: 400; }
.kpi-note { font-size: 12.5px; color: var(--ink-3); margin-top: 6px; }

.callout {
  border-left: 2px solid var(--accent);
  background: var(--accent-soft);
  padding: 14px 18px;
  margin: 22px 0;
  border-radius: 0 4px 4px 0;
}
.callout p { margin: 0; max-width: 64ch; font-size: 15px; }
.callout p + p { margin-top: 10px; }

.defect {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 18px 20px;
  margin-bottom: 14px;
}
.defect h3 { margin: 0 0 8px; font-size: 15.5px; }
.defect p { margin: 0 0 10px; font-size: 14.5px; }
.defect p:last-child { margin-bottom: 0; }

.tag {
  display: inline-block;
  font-size: 10.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid currentColor;
  margin-bottom: 10px;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
}
.tag-bug { color: var(--accent); }
.tag-gap { color: var(--ink-3); }
.tag-fixed { color: var(--good); }

footer {
  margin-top: 72px;
  padding-top: 22px;
  border-top: 1px solid var(--rule);
  font-size: 13px;
  color: var(--ink-3);
}

@media (max-width: 640px) {
  .wrap { padding: 0 18px 72px; }
  .masthead { padding-top: 40px; }
  section { padding-top: 46px; }
  .figure { padding: 16px 14px 12px; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""
