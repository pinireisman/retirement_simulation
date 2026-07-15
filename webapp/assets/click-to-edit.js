/* dash_table cell editing made to behave like a normal text box.
   All findings below verified empirically against this dash version.

   Problems fixed:
   1. A single click leaves the cell in a half-edit state ("input-active
      unfocused"): value invisibly select-all'd, Backspace wipes the cell,
      typing replaces everything, arrows jump to neighboring cells.
      -> promote the click to real edit mode, caret at the clicked character.
   2. Real edit mode DISCARDS the value on click-away (only Enter/Tab commit).
      -> dispatch a synthetic Enter on mousedown outside the editing cell.
   3. dash resets the selection (select-all) on any mouse event inside an
      editing input and hijacks Ctrl+C/V at the document level (cell-range
      copy/paste), so caret clicks, drag-select, and clipboard don't work.
      -> stop propagation of mouse + clipboard events on the editing input
         so the browser's native caret/selection/clipboard behavior runs.
         Cell-range copy while NOT editing is untouched.

   Only mouse clicks are promoted — arrow-key navigation between cells also
   produces the "unfocused" input state and must stay in navigation mode. */

(() => {
  const editingInput = () => {
    const el = document.activeElement;
    return el && el.tagName === "INPUT" &&
           el.classList.contains("dash-cell-value") &&
           el.classList.contains("focused") ? el : null;
  };

  /* ---- 3. isolate the editing input from dash's global handlers ---- */
  for (const type of ["mousedown", "mouseup", "click", "dblclick"]) {
    document.addEventListener(type, (e) => {
      const inp = editingInput();
      if (inp && e.target === inp) {
        e.stopPropagation();
        pending = null; // user is placing the caret/selection directly
      }
    }, true);
  }
  document.addEventListener("keydown", (e) => {
    if (!editingInput()) return;
    if ((e.ctrlKey || e.metaKey) &&
        ["c", "v", "x", "a", "z"].includes(e.key.toLowerCase())) {
      e.stopPropagation(); // keep dash's cell-range copy/paste out; native wins
    }
  }, true);
  for (const type of ["copy", "paste", "cut"]) {
    document.addEventListener(type, (e) => {
      if (editingInput() === e.target) e.stopPropagation();
    }, true);
  }

  /* ---- 1. promote single click to edit mode, caret at clicked char ---- */

  // Caret offset captured at mousedown, while the cell still renders its
  // display text as a plain text node (after promotion it's an <input>,
  // which caretRangeFromPoint can't see into).
  let clickOffset = null; // {td, offset}
  let pending = null;     // {inp, pos} — placement being defended

  document.addEventListener("mousedown", (e) => {
    clickOffset = null;
    const td = e.target.closest && e.target.closest("td.dash-cell");
    if (!td) return;
    const pos = document.caretRangeFromPoint
      ? document.caretRangeFromPoint(e.clientX, e.clientY)
      : (document.caretPositionFromPoint &&
         document.caretPositionFromPoint(e.clientX, e.clientY));
    const node = pos && (pos.startContainer || pos.offsetNode);
    if (node && node.nodeType === Node.TEXT_NODE && td.contains(node)) {
      clickOffset = { display: node.textContent,
                      offset: pos.startOffset !== undefined ? pos.startOffset : pos.offset };
    }
  }, true);

  // Display text can be formatted ("15,000") while the input holds the raw
  // value ("15000") — walk both to translate the offset.
  const mapOffset = (display, dispOff, value) => {
    let j = 0;
    for (let i = 0; i < dispOff && i < display.length; i++) {
      if (j < value.length && display[i] === value[j]) j++;
    }
    return j;
  };

  const place = () => {
    if (!pending) return;
    // Re-resolve the input: dash can remount it when the unfocused->focused
    // flip lands, so an element pinned at promotion time may be stale.
    const inp = editingInput();
    if (!inp) return;
    const pos = Math.min(pending.pos, inp.value.length);
    // Only (re)apply while dash's select-all reset is what's there — never
    // fight a caret/selection the user has since moved.
    const untouched = (inp.selectionStart === 0 && inp.selectionEnd === inp.value.length) ||
                      (inp.selectionStart === pos && inp.selectionEnd === pos);
    if (untouched) inp.setSelectionRange(pos, pos);
  };

  // Resolve the cell from the focused input each attempt, never from the
  // click target: dash may re-render the table (detaching nodes) between
  // mousedown and here, and it can also swallow a dblclick dispatched while
  // it is mid-update — hence the retries.
  const promote = (captured, tries, dispatched) => {
    const el = document.activeElement;
    if (!el || el.tagName !== "INPUT" ||
        !el.classList.contains("dash-cell-value")) return;
    if (el.classList.contains("unfocused")) {
      const td = el.closest("td.dash-cell");
      if (td) td.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
      dispatched = true;
    }
    const inp = document.activeElement;
    if (!inp || inp.tagName !== "INPUT" ||
        !inp.classList.contains("dash-cell-value")) return;
    if (!inp.classList.contains("focused") || inp.classList.contains("unfocused")) {
      if (tries > 0) setTimeout(() => promote(captured, tries - 1, dispatched), 50);
      return;
    }
    // Focused without us dispatching = the user really double-clicked and
    // dash's own select-all is intentional — leave it alone.
    if (!dispatched) return;
    const pos = captured
      ? mapOffset(captured.display, captured.offset, inp.value)
      : inp.value.length;
    pending = { pos };
    requestAnimationFrame(place);
    // dash's async select-all can land after the first placement — defend
    // briefly, then let go.
    setTimeout(place, 150);
    setTimeout(() => { place(); pending = null; }, 350);
  };

  document.addEventListener("click", (e) => {
    if (!(e.target.closest && e.target.closest("td.dash-cell"))) return;
    // The offset was captured by this click's own mousedown moments ago.
    const captured = clickOffset;
    clickOffset = null;
    // Let dash's own click handling finish first (it mounts the input).
    setTimeout(() => promote(captured, 10, false), 0);
    // capture phase: dash stops propagation of some cell clicks (e.g. the
    // first click after an Enter commit), which would starve a bubble listener
  }, true);

  // A real double-click means "select all" — don't fight it.
  document.addEventListener("dblclick", (e) => {
    if (e.isTrusted) pending = null;
  }, true);

  /* ---- 2. commit on click-away (spreadsheet convention) ---- */
  document.addEventListener("mousedown", (e) => {
    const el = editingInput();
    if (!el) return;
    const td = el.closest("td.dash-cell");
    if (!td || td.contains(e.target)) return;
    const enter = new KeyboardEvent("keydown", { key: "Enter", code: "Enter", bubbles: true });
    Object.defineProperty(enter, "keyCode", { get: () => 13 });
    Object.defineProperty(enter, "which", { get: () => 13 });
    el.dispatchEvent(enter);
  }, true);
})();
