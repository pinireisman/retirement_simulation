document.addEventListener("click", function (e) {
  var btn = e.target.closest(".btn-maximize");
  if (!btn) return;
  var card = btn.closest(".chart-card");
  if (!card) return;
  card.classList.toggle("maximized");
  setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 50);
});

document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") {
    var opened = document.querySelectorAll(".chart-card.maximized");
    if (!opened.length) return;
    opened.forEach(function (card) { card.classList.remove("maximized"); });
    setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 50);
  }
});
