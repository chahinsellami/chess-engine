const PROMOTIONS = { q: "Q", r: "R", b: "B", n: "N" };
let state = null;
let selected = null;
let pendingPromotion = null;
let pollTimer = null;

const $ = (id) => document.getElementById(id);
const files = "abcdefgh";
const pieceAt = (fen, square) => {
  const rows = fen.split(" ")[0].split("/");
  const rank = 8 - Number(square[1]);
  let file = 0;
  for (const char of rows[rank]) {
    if (Number.isInteger(Number(char))) file += Number(char);
    else {
      if (files[file] === square[0]) return char;
      file += 1;
    }
  }
  return null;
};
const squareFromIndex = (index) => `${files[index % 8]}${8 - Math.floor(index / 8)}`;
const showToast = (message) => {
  $("toast").textContent = message;
  $("toast").classList.add("show");
  setTimeout(() => $("toast").classList.remove("show"), 2600);
};
async function api(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}
function renderBoard() {
  const board = $("board");
  board.innerHTML = "";
  const legal = new Set(state.legal_moves);
  for (let index = 0; index < 64; index += 1) {
    const square = squareFromIndex(index);
    const piece = pieceAt(state.fen, square);
    const button = document.createElement("button");
    const isLight = (index + Math.floor(index / 8)) % 2 === 0;
    button.className = `square ${isLight ? "light" : "dark"}`;
    if (state.last_move && [state.last_move.slice(0, 2), state.last_move.slice(2, 4)].includes(square)) button.classList.add("last");
    if (selected === square) button.classList.add("selected");
    if (selected && legal.has(`${selected}${square}`)) button.classList.add(piece ? "capture" : "legal");
    button.dataset.square = square;
    button.setAttribute("aria-label", `${square}${piece ? ` ${piece}` : ""}`);
    button.addEventListener("click", () => selectSquare(square));
    if (piece) {
      const image = document.createElement("img");
      image.className = `piece ${piece === piece.toUpperCase() ? "white" : "black"}`;
      image.src = `/static/pieces/${piece === piece.toUpperCase() ? "w" : "b"}${piece.toUpperCase()}.svg`;
      image.alt = `${piece === piece.toUpperCase() ? "White" : "Black"} ${piece.toUpperCase()}`;
      button.appendChild(image);
    }
    if (index % 8 === 0) { const rank = document.createElement("span"); rank.className = "coord rank"; rank.textContent = square[1]; button.appendChild(rank); }
    if (index > 55) { const file = document.createElement("span"); file.className = "coord file"; file.textContent = square[0]; button.appendChild(file); }
    board.appendChild(button);
  }
}
function selectSquare(square) {
  if (!state || state.thinking || state.turn !== "white" || state.game_over) return;
  const legal = new Set(state.legal_moves);
  const piece = pieceAt(state.fen, square);
  if (!selected) {
    if (piece && piece === piece.toUpperCase()) { selected = square; renderBoard(); }
    return;
  }
  if (selected === square) { selected = null; renderBoard(); return; }
  const base = `${selected}${square}`;
  if (legal.has(base)) {
    const movingPiece = pieceAt(state.fen, selected);
    if (movingPiece === "P" && square[1] === "8") openPromotion(base, "white");
    else submitMove(base);
  } else if (piece && piece === piece.toUpperCase()) { selected = square; renderBoard(); }
  else { selected = null; renderBoard(); }
}
function openPromotion(base) {
  pendingPromotion = base;
  $("promotionChoices").innerHTML = "";
  Object.entries(PROMOTIONS).forEach(([letter, piece]) => {
    const button = document.createElement("button");
    const image = document.createElement("img");
    image.src = `/static/pieces/w${piece}.svg`;
    image.alt = `Promote to ${letter}`;
    button.appendChild(image);
    button.setAttribute("aria-label", `Promote to ${letter}`);
    button.onclick = () => { $("promotionModal").classList.add("hidden"); submitMove(`${pendingPromotion}${letter}`); };
    $("promotionChoices").appendChild(button);
  });
  $("promotionModal").classList.remove("hidden");
}
async function submitMove(uci) {
  selected = null; renderBoard();
  try { state = await api("/api/move", { method: "POST", body: JSON.stringify({ uci }) }); render(); startPolling(); }
  catch (error) { showToast(error.message); await refresh(); }
}
function render() {
  renderBoard();
  $("statusTitle").textContent = state.status;
  $("statusDetail").textContent = state.thinking ? "The neural engine is searching the position." : state.game_over ? "This game has reached a final position." : "Choose a move on the board to continue.";
  $("turnBadge").textContent = state.thinking ? "ENGINE THINKING" : state.status.toUpperCase();
  $("evalValue").textContent = `${state.evaluation >= 0 ? "+" : ""}${state.evaluation.toFixed(2)}`;
  $("evalBar").style.width = `${Math.max(8, Math.min(92, 50 + state.evaluation * 8))}%`;
  $("analysisCopy").textContent = state.search_error || (state.engine_ready ? "Neural ordering + alpha-beta search are active." : "Running in classical fallback mode; see Validation.");
  $("moveCount").textContent = `${String(state.history.length).padStart(2, "0")} MOVES`;
  const list = $("moveList"); list.innerHTML = "";
  if (!state.history.length) list.innerHTML = '<div class="empty-state">Your first move is waiting.</div>';
  for (let i = 0; i < state.history.length; i += 2) {
    const row = document.createElement("div"); row.className = "move-row";
    row.innerHTML = `<span>${String(i / 2 + 1).padStart(2, "0")}</span><span>${state.history[i]?.san || ""}</span><span>${state.history[i + 1]?.san || ""}</span>`;
    list.appendChild(row);
  }
}
async function refresh() { try { state = await api("/api/state"); render(); if (state.thinking) startPolling(); } catch (error) { showToast(error.message); } }
function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => { await refresh(); if (!state?.thinking) clearInterval(pollTimer); }, 700);
}
async function loadValidation() {
  $("validationTitle").textContent = "Checking system";
  const report = await api("/api/validation");
  $("validationTitle").textContent = `${report.passed}/${report.total} checks passed`;
  $("validationSummary").textContent = report.passed === report.total ? "The engine stack is ready for a clean run." : "One or more checks need attention before training or play.";
  $("checkList").innerHTML = report.checks.map((check) => `<div class="check ${check.ok ? "ok" : ""}"><span class="check-icon">${check.ok ? "✓" : "!"}</span><div><strong>${check.name}</strong><small>${check.detail}</small></div><span class="check-state">${check.ok ? "PASS" : "REVIEW"}</span></div>`).join("");
  const training = report.training;
  $("valAccuracy").textContent = training.metrics_available && training.validation_accuracy != null ? `${Number(training.validation_accuracy).toFixed(1)}%` : "—";
  $("valTop5").textContent = training.metrics_available && training.validation_top5 != null ? `${Number(training.validation_top5).toFixed(1)}%` : "—";
  $("valEpoch").textContent = training.metrics_available ? training.epoch : "—";
  $("metricSource").textContent = training.metrics_available ? `Source: ${training.source}` : "Validation metrics not recorded in the checkpoint.";
}
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === tab));
  $("gamePanel").classList.toggle("hidden", tab.dataset.tab !== "game");
  $("validationPanel").classList.toggle("hidden", tab.dataset.tab !== "validation");
  if (tab.dataset.tab === "validation") loadValidation().catch((error) => showToast(error.message));
}));
$("newGame").onclick = async () => { try { state = await api("/api/new", { method: "POST" }); selected = null; render(); } catch (error) { showToast(error.message); } };
$("undo").onclick = async () => { try { state = await api("/api/undo", { method: "POST" }); selected = null; render(); } catch (error) { showToast(error.message); } };
$("refreshValidation").onclick = () => loadValidation().catch((error) => showToast(error.message));
refresh();
