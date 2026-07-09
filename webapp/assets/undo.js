document.addEventListener("keydown", function (e) {
  if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
    var btn = document.getElementById("btn-undo");
    if (btn && !btn.disabled) {
      e.preventDefault();
      btn.click();
    }
  }
});
