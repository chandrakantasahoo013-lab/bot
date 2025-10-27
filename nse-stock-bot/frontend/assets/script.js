let charts = {}; let allStocks = [];

async function updateDashboard() {
    try {
        const response = await fetch('http://localhost:5001/status'); // Updated to match new backend port
        const data = await response.json();
        allStocks = data.stocks;
        document.getElementById('market-status').innerHTML = `<strong>Market Open:</strong> ${data.market_open ? '<span class="text-success">Yes</span>' : '<span class="text-danger">No</span>'} (NSE Hours)`;
        renderStocks(allStocks);
        document.getElementById('last-update').textContent = new Date(data.last_check).toLocaleString('en-IN');
    } catch (error) { console.error('Error:', error); }
}

function renderStocks(stocks) {
    const grid = document.getElementById('stocks-grid');
    grid.innerHTML = stocks.map(stock => `
        <div class="col-lg-4 col-md-6 mb-4 fade-in">
            <div class="card h-100">
                <div class="card-header bg-gradient text-white d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">${stock.symbol}</h5>
                    <span class="badge bg-light text-dark">${stock.position}</span>
                </div>
                <div class="card-body">
                    <p class="fw-bold fs-5 text-center mb-3">₹${stock.price.toFixed(2)}</p>
                    <div class="row mb-3">
                        <div class="col-4"><strong>Combined:</strong> <span class="badge signal-badge ${stock.combined_signal.toLowerCase()}">${stock.combined_signal}</span></div>
                        <div class="col-4"><strong>Long:</strong> <span class="badge signal-badge ${stock.long_signal.toLowerCase()}">${stock.long_signal}</span></div>
                        <div class="col-4"><strong>Short:</strong> <span class="badge signal-badge ${stock.short_signal.toLowerCase()}">${stock.short_signal}</span></div>
                    </div>
                    <p class="mb-2"><strong>Sentiment:</strong> ${stock.sentiment > 0.5 ? '👍' : stock.sentiment < -0.5 ? '👎' : '😐'} ${stock.sentiment}</p>
                    <p class="mb-3"><strong>Capital:</strong> ₹${stock.capital.toFixed(2)}</p>
                    <div class="chart-container"><canvas id="chart-${stock.symbol}"></canvas></div>
                    <div class="accordion mt-3" id="news-${stock.symbol}">
                        <div class="accordion-item"><h2 class="accordion-header"><button class="accordion-button collapsed fw-semibold" data-bs-toggle="collapse" data-bs-target="#collapse-${stock.symbol}">📢 Recent News (${stock.news.length})</button></h2>
                            <div id="collapse-${stock.symbol}" class="accordion-collapse collapse"><div class="accordion-body">${stock.news.map(n => `<div class="news-item news-${n.type} p-2 rounded mb-2">${n.text}</div>`).join('')}</div></div>
                        </div>
                    </div>
                    <div class="accordion mt-3" id="indicators-${stock.symbol}">
                        <div class="accordion-item"><h2 class="accordion-header"><button class="accordion-button collapsed fw-semibold" data-bs-toggle="collapse" data-bs-target="#ind-collapse-${stock.symbol}">▼ Indicators</button></h2>
                            <div id="ind-collapse-${stock.symbol}" class="accordion-collapse collapse"><div class="accordion-body">
                                <ul class="list-unstyled">${stock.insights.map(i => `<li class="text-muted">${i}</li>`).join('')}</ul>
                            </div></div>
                        </div>
                    </div>
                    <div class="mt-3">
                        <button class="btn btn-success me-2" onclick="simulateTrade('${stock.symbol}', 'buy')">Sim Buy</button>
                        <button class="btn btn-danger" onclick="simulateTrade('${stock.symbol}', 'sell')">Sim Sell</button>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
    stocks.forEach(stock => {
        const ctx = document.getElementById(`chart-${stock.symbol}`).getContext('2d');
        if (ctx && stock.recent_prices) {
            if (charts[stock.symbol]) charts[stock.symbol].destroy();
            charts[stock.symbol] = new Chart(ctx, {
                type: 'line', data: { labels: stock.price_labels, datasets: [{ label: 'Price', data: stock.recent_prices, borderColor: '#0d6efd', backgroundColor: 'rgba(13,110,253,0.1)', tension: 0.4, fill: true }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        }
    });
}

function filterStocks() {
    const query = document.getElementById('search-input').value.toLowerCase();
    renderStocks(allStocks.filter(s => s.symbol.toLowerCase().includes(query)));
}

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    document.getElementById('theme-toggle').textContent = document.body.classList.contains('dark-mode') ? '☀️ Light Mode' : '🌙 Dark Mode';
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
}

function simulateTrade(symbol, side) {
    fetch('http://localhost:5001/trade', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol, side }) })
        .then(response => response.json()).then(data => { if (data.status === 'success') updateDashboard(); });
}

function addStock() { alert('Add not implemented yet—edit symbols in backend.'); }
function removeStock() { alert('Remove not implemented yet—edit symbols in backend.'); }

if (localStorage.getItem('darkMode') === 'true') toggleDarkMode();
updateDashboard();
setInterval(updateDashboard, 300000);