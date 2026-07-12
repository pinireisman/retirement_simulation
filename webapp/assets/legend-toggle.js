document.addEventListener("click", function (e) {
  var btn = e.target.closest(".btn-legend-toggle");
  if (!btn) return;
  var card = btn.closest(".chart-card");
  var gd = card && card.querySelector(".js-plotly-plot");
  if (!gd || !window.Plotly) return;
  if (gd._legendBaseHeight === undefined) gd._legendBaseHeight = gd.layout.height;
  var showing = !gd.layout.showlegend;
  var legendHeight = (gd.layout.meta && gd.layout.meta.legendHeight) || gd._legendBaseHeight;
  window.Plotly.relayout(gd, {
    showlegend: showing,
    height: showing ? legendHeight : gd._legendBaseHeight,
  });
});
