"use strict";

const count = (value) => new Intl.NumberFormat("zh-CN").format(value);
const percent = (ratio) => (ratio * 100).toFixed(1) + "%";
const name = (value) => value.toUpperCase();
const get = (id) => document.getElementById(id);

function node(tag, value, className) {
  const element = document.createElement(tag);
  if (value !== undefined) element.textContent = value;
  if (className) element.className = className;
  return element;
}

function heat(rate) {
  const stops = [[91, 49, 87], [38, 64, 100], [23, 124, 112]];
  const index = rate < 0.5 ? 0 : 1;
  const fraction = 2 * rate - index;
  const rgb = stops[index].map((start, channel) => Math.round(start + (stops[index + 1][channel] - start) * fraction));
  return "rgb(" + rgb.join(",") + ")";
}

function counts(row, prefix = "") {
  return ["wins", "losses", "draws"].map((key) => count(row[prefix + key])).join(" / ");
}

let selected;
function showMatch(cell, button) {
  if (selected) selected.setAttribute("aria-pressed", "false");
  selected = button;
  selected.setAttribute("aria-pressed", "true");
  get("match-details").replaceChildren(...[
    ["对阵", name(cell.policy) + " vs " + name(cell.opponent)],
    ["合计 · 胜／负／平", counts(cell)],
    ["行策略为红方", counts(cell, "red_")],
    ["行策略为蓝方", counts(cell, "blue_")],
  ].map(([label, value]) => {
    const item = node("div");
    item.append(node("dt", label), node("dd", value));
    return item;
  }));
}

function tableHeader(label, scope) {
  const header = node("th", label);
  header.scope = scope;
  return header;
}

function renderMatrix(report) {
  const byMatch = new Map(report.cells.map((cell) => [cell.policy + "/" + cell.opponent, cell]));
  const table = get("matrix-table");
  const head = table.createTHead().insertRow();
  head.append(tableHeader("行 / 列", "col"), ...report.policies.map((policy) => tableHeader(name(policy), "col")));
  const body = table.createTBody();
  for (const policy of report.policies) {
    const row = body.insertRow();
    row.append(tableHeader(name(policy), "row"));
    for (const opponent of report.policies) {
      const cell = row.insertCell();
      if (policy === opponent) {
        cell.append(node("span", "—", "b01-diagonal"));
        continue;
      }
      const matchup = byMatch.get(policy + "/" + opponent);
      const button = node("button", percent(matchup.win_rate), "b01-cell");
      button.type = "button";
      button.style.setProperty("--heat", heat(matchup.win_rate));
      button.setAttribute("aria-label", name(policy) + " 对 " + name(opponent) + " " + percent(matchup.win_rate));
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => showMatch(matchup, button));
      cell.append(button);
      if (!selected) showMatch(matchup, button);
    }
  }
  get("matrix-caption").textContent =
    "行策略对列策略的胜率。每对 " + count(report.layouts * 2) + " 局，平局计入分母。共 "
    + count(report.pairwise_games) + " 局两两对抗。选择单元格可查看红蓝明细。";
}

function renderRanking(report) {
  get("ranking-list").replaceChildren(...report.ranking.map((entry) => {
    const item = node("li");
    const bar = node("span", undefined, "rank-bar");
    const fill = node("i");
    fill.style.width = percent(entry.mean_win_rate);
    bar.append(fill);
    bar.setAttribute("aria-hidden", "true");
    item.append(node("span", name(entry.policy)), bar, node("strong", percent(entry.mean_win_rate)));
    return item;
  }));
}

function renderSelfPlay(report) {
  get("self-play-rows").replaceChildren(...report.self_play.map((entry) => {
    const tr = node("tr");
    const delta = 100 * (entry.red_wins - entry.blue_wins) / entry.games;
    tr.append(tableHeader(name(entry.policy), "row"),
      ...[entry.games, entry.red_wins, entry.blue_wins, entry.draws].map((value) => node("td", count(value))),
      node("td", (delta > 0 ? "+" : "") + delta.toFixed(1) + " 个百分点"));
    return tr;
  }));
}

function renderBot(report) {
  get("bot-caption").textContent = "冻结策略固定红方，原生 Bot 固定蓝方，难度 " + report.bot_difficulty
    + "。每种策略 " + count(report.layouts) + " 局，与主矩阵共用出生点。";
  get("bot-rows").replaceChildren(...report.rows.map((entry) => {
    const tr = node("tr");
    tr.append(tableHeader(name(entry.policy), "row"), node("td", percent(entry.win_rate)),
      ...[entry.wins, entry.losses, entry.draws, entry.games].map((value) => node("td", count(value))));
    return tr;
  }));
}

async function load() {
  const response = await fetch("./data/report.json");
  if (!response.ok) throw new Error("HTTP " + response.status);
  const report = await response.json();
  renderMatrix(report);
  renderRanking(report);
  renderSelfPlay(report);
  const bot = await fetch("./data/bot-report.json");
  if (!bot.ok) throw new Error("Bot HTTP " + bot.status);
  renderBot(await bot.json());
  const target = get(location.hash.slice(1));
  if (target) requestAnimationFrame(() => target.scrollIntoView({ block: "start" }));
}

load().catch((error) => {
  const feedback = get("matrix-error");
  feedback.hidden = false;
  feedback.textContent = "B01 结果暂不可用（" + error.message + "）。";
});
