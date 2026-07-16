/* Drag dash_table rows by the ⠿ handle column to reorder.

   build_data_table prepends a data-less "drag-handle" column (glyph drawn by
   style.css). pointerdown on a handle cell starts the drag immediately:
   a fixed-position ghost copy of the row follows the cursor, and the real
   row (dimmed as a placeholder) is live-reordered in the DOM with
   swap-on-enter semantics. On drop the DOM move is reverted (the tbody is
   React-owned; leaving it mutated breaks reconciliation) and {table, from,
   to} is pushed into the store-row-move dcc.Store via
   dash_clientside.set_props; a server callback reorders the table's `data`
   prop and React re-renders the new order.

   Canceling pointerdown suppresses the compatibility mousedown, which keeps
   dash's cell-selection (and click-to-edit.js) off the handle cell. */

(() => {
  let drag = null;       // {tr, tableId, from, ghost}
  let seq = 0;
  let suppressClick = false;

  const HANDLE = 'td.dash-cell[data-dash-column="drag-handle"]';

  const dataRows = (tbody) =>
    Array.from(tbody.children).filter((r) => r.querySelector("td.dash-cell"));

  const rowAt = (x, y) => {
    const el = document.elementFromPoint(x, y);
    const tr = el && el.closest && el.closest("tr");
    return tr && tr.querySelector("td.dash-cell") ? tr : null;
  };

  const makeGhost = (tr, y) => {
    const rect = tr.getBoundingClientRect();
    const ghost = document.createElement("table");
    ghost.className = "row-drag-ghost";
    const tbody = document.createElement("tbody");
    const clone = tr.cloneNode(true);
    // freeze column widths and alignment — the ghost table has no header to
    // size against and dash's cell CSS doesn't reach outside the container
    Array.from(tr.children).forEach((cell, i) => {
      const cs = getComputedStyle(cell);
      Object.assign(clone.children[i].style, {
        width: cell.getBoundingClientRect().width + "px",
        boxSizing: "border-box",
        textAlign: cs.textAlign,
        padding: cs.padding,
      });
    });
    tbody.appendChild(clone);
    ghost.appendChild(tbody);
    ghost.style.left = rect.left + "px";
    ghost.style.top = (y - rect.height / 2) + "px";
    document.body.appendChild(ghost);
    return ghost;
  };

  const endDrag = () => {
    if (!drag) return;
    drag.ghost.remove();
    drag.tr.classList.remove("row-drag-placeholder");
    document.body.classList.remove("row-drag-active");
    drag = null;
  };

  document.addEventListener("pointerdown", (e) => {
    if (!e.isPrimary || e.button !== 0) return;
    const handle = e.target.closest && e.target.closest(HANDLE);
    if (!handle) return;
    const tr = handle.closest("tr");
    const container = tr && tr.closest(".dash-table-container");
    if (!tr || !container || !container.id) return;
    e.preventDefault();
    e.stopPropagation();
    drag = {
      tr,
      tableId: container.id,
      from: dataRows(tr.parentNode).indexOf(tr),
      ghost: makeGhost(tr, e.clientY),
    };
    tr.classList.add("row-drag-placeholder");
    document.body.classList.add("row-drag-active");
  }, true);

  // belt for browsers that fire mousedown despite the canceled pointerdown
  document.addEventListener("mousedown", (e) => {
    if (e.target.closest && e.target.closest(HANDLE)) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, true);

  document.addEventListener("pointermove", (e) => {
    if (!drag) return;
    e.preventDefault();
    drag.ghost.style.top =
      (e.clientY - drag.ghost.getBoundingClientRect().height / 2) + "px";
    // probe at the table's horizontal center so leaving the handle column
    // (or the table edge) sideways doesn't stall the reorder
    const ref = drag.tr.getBoundingClientRect();
    const target = rowAt(ref.left + ref.width / 2, e.clientY);
    if (!target || target === drag.tr ||
        target.parentNode !== drag.tr.parentNode) return;
    // swap on enter (sortable.js semantics): moving down puts the dragged
    // row after the row entered, moving up puts it before
    const below = drag.tr.compareDocumentPosition(target) & Node.DOCUMENT_POSITION_FOLLOWING;
    target.parentNode.insertBefore(drag.tr, below ? target.nextSibling : target);
  }, true);

  document.addEventListener("pointerup", () => {
    if (!drag) return;
    const { tr, tableId, from } = drag;
    const rows = dataRows(tr.parentNode);
    const to = rows.indexOf(tr);
    endDrag();
    suppressClick = true; // the release still fires a click; keep it off dash
    if (to === from || from < 0 || to < 0) return;
    // revert the feedback move: React owns this tbody and re-renders from data
    rows.splice(to, 1);
    tr.parentNode.insertBefore(tr, rows[from] || null);
    if (window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props("store-row-move",
        { data: { table: tableId, from, to, seq: ++seq } });
    }
  }, true);

  document.addEventListener("pointercancel", endDrag, true);

  document.addEventListener("click", (e) => {
    if (suppressClick) {
      suppressClick = false;
      e.stopPropagation();
      e.preventDefault();
    }
  }, true);
})();
