document.addEventListener("click", function (e) {
  const btn = e.target.closest(".btn");
  if (!btn) return;
  const rect = btn.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  const span = document.createElement("span");
  span.className = "ripple";
  span.style.width = span.style.height = size + "px";
  span.style.left = (e.clientX - rect.left - size / 2) + "px";
  span.style.top = (e.clientY - rect.top - size / 2) + "px";
  btn.appendChild(span);
  span.addEventListener("animationend", () => span.remove());
});
