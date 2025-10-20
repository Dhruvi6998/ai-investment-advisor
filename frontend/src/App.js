import React, { useState, useEffect } from 'react';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, PieChart as PieIcon, RefreshCw, LogOut, Plus, Trash2, AlertCircle, CheckCircle, Eye, EyeOff } from 'lucide-react';

const API_URL = 'http://localhost:5000/api';

export default function AIInvestmentAdvisor() {
  const [auth, setAuth] = useState({ isAuthenticated: false, username: null });
  const [authMode, setAuthMode] = useState('login');
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [showPassword, setShowPassword] = useState(false);
  
  const [portfolio, setPortfolio] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [rebalancing, setRebalancing] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const [newStock, setNewStock] = useState({ ticker: '', quantity: '', purchase_price: '' });
  const [searchResult, setSearchResult] = useState(null);
  const [activeTab, setActiveTab] = useState('portfolio');
  
  const [notification, setNotification] = useState(null);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const response = await fetch(`${API_URL}/check-auth`, {
        credentials: 'include'
      });
      const data = await response.json();
      if (data.authenticated) {
        setAuth({ isAuthenticated: true, username: data.username });
        loadPortfolio();
      }
    } catch (error) {
      console.error('Auth check failed:', error);
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const endpoint = authMode === 'login' ? '/login' : '/register';
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(credentials)
      });
      const data = await response.json();
      
      if (data.success) {
        if (authMode === 'register') {
          showNotification('Registration successful! Please login.', 'success');
          setAuthMode('login');
        } else {
          setAuth({ isAuthenticated: true, username: data.username });
          showNotification('Welcome back!', 'success');
          loadPortfolio();
        }
        setCredentials({ username: '', password: '' });
      } else {
        showNotification(data.message, 'error');
      }
    } catch (error) {
      showNotification('Authentication failed', 'error');
    }
    setLoading(false);
  };

  const handleLogout = async () => {
    await fetch(`${API_URL}/logout`, { method: 'POST', credentials: 'include' });
    setAuth({ isAuthenticated: false, username: null });
    setPortfolio([]);
    setAnalysis(null);
    showNotification('Logged out successfully', 'success');
  };

  const loadPortfolio = async () => {
    try {
      const response = await fetch(`${API_URL}/portfolio`, { credentials: 'include' });
      const data = await response.json();
      if (data.success) {
        setPortfolio(data.portfolio);
      }
    } catch (error) {
      console.error('Failed to load portfolio:', error);
    }
  };

  const searchStock = async () => {
    if (!newStock.ticker) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/stock-search?ticker=${newStock.ticker}`, {
        credentials: 'include'
      });
      const data = await response.json();
      if (data.success) {
        setSearchResult(data.stock);
        showNotification(`Found: ${data.stock.name}`, 'success');
      } else {
        showNotification('Stock not found', 'error');
        setSearchResult(null);
      }
    } catch (error) {
      showNotification('Search failed', 'error');
    }
    setLoading(false);
  };

  const addStock = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/portfolio/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(newStock)
      });
      const data = await response.json();
      
      if (data.success) {
        showNotification('Stock added successfully!', 'success');
        setNewStock({ ticker: '', quantity: '', purchase_price: '' });
        setSearchResult(null);
        loadPortfolio();
        setActiveTab('portfolio');
      } else {
        showNotification(data.message, 'error');
      }
    } catch (error) {
      showNotification('Failed to add stock', 'error');
    }
    setLoading(false);
  };

  const removeStock = async (ticker) => {
    try {
      const response = await fetch(`${API_URL}/portfolio/remove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ ticker })
      });
      const data = await response.json();
      
      if (data.success) {
        showNotification('Stock removed', 'success');
        loadPortfolio();
        if (analysis) runAnalysis();
      }
    } catch (error) {
      showNotification('Failed to remove stock', 'error');
    }
  };

  const runAnalysis = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/analyze`, { credentials: 'include' });
      const data = await response.json();
      if (data.success) {
        setAnalysis(data.analysis);
        showNotification('Analysis complete!', 'success');
      }
    } catch (error) {
      showNotification('Analysis failed', 'error');
    }
    setLoading(false);
  };

  const runRebalancing = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/rebalance`, { credentials: 'include' });
      const data = await response.json();
      console.log('Full rebalancing response:', data); // Debug
      console.log('Rebalancing data:', data.rebalancing); // Debug
      
      if (data.success && data.rebalancing) {
        setRebalancing(data.rebalancing);
        showNotification('Rebalancing analysis complete!', 'success');
      } else if (data.success && data.rebalancing === false) {
        // Handle case where rebalancing_needed is false
        setRebalancing({ rebalancing_needed: false, message: 'Need at least 2 stocks' });
        showNotification('Not enough stocks for rebalancing', 'error');
      } else {
        showNotification(data.message || 'Rebalancing analysis failed', 'error');
      }
    } catch (error) {
      console.error('Rebalancing error:', error);
      showNotification('Rebalancing analysis failed', 'error');
    }
    setLoading(false);
  };

  const showNotification = (message, type) => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 4000);
  };

  const getActionColor = (action) => {
    switch (action) {
      case 'STRONG_SELL': return 'text-white bg-red-700 border-red-800 shadow-lg';
      case 'SELL': return 'text-white bg-red-600 border-red-700 shadow-md';
      case 'CONSIDER_SELLING': return 'text-white bg-orange-500 border-orange-600';
      case 'HOLD': return 'text-gray-700 bg-gray-200 border-gray-300';
      case 'HOLD_OR_BUY': return 'text-blue-700 bg-blue-100 border-blue-300';
      case 'BUY': return 'text-green-700 bg-green-100 border-green-300';
      case 'STRONG_BUY': return 'text-white bg-green-600 border-green-700 shadow-md';
      default: return 'text-gray-600 bg-gray-100 border-gray-200';
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'Very High': return 'bg-red-600 text-white';
      case 'High': return 'bg-red-500 text-white';
      case 'Moderate': return 'bg-yellow-500 text-white';
      case 'Low': return 'bg-green-500 text-white';
      case 'Very Low': return 'bg-green-600 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  const getPerformanceColor = (value) => {
    if (value >= 20) return 'text-green-600 bg-green-50 border-green-200';
    if (value >= 10) return 'text-green-500 bg-green-50 border-green-100';
    if (value >= 5) return 'text-blue-500 bg-blue-50 border-blue-100';
    if (value >= 0) return 'text-gray-500 bg-gray-50 border-gray-100';
    if (value >= -5) return 'text-orange-500 bg-orange-50 border-orange-100';
    if (value >= -10) return 'text-red-500 bg-red-50 border-red-100';
    return 'text-red-600 bg-red-100 border-red-200';
  };

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

  if (!auth.isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <TrendingUp className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-gray-900">AI Investment Advisor</h1>
            <p className="text-gray-600 mt-2">Smart portfolio management powered by AI</p>
          </div>

          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setAuthMode('login')}
              className={`flex-1 py-2 px-4 rounded-lg font-medium transition ${
                authMode === 'login'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Login
            </button>
            <button
              onClick={() => setAuthMode('register')}
              className={`flex-1 py-2 px-4 rounded-lg font-medium transition ${
                authMode === 'register'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Register
            </button>
          </div>

          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Username</label>
              <input
                type="text"
                value={credentials.username}
                onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={credentials.password}
                  onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 transition disabled:opacity-50"
            >
              {loading ? 'Processing...' : authMode === 'login' ? 'Login' : 'Register'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {notification && (
        <div className={`fixed top-4 right-4 z-50 px-6 py-4 rounded-lg shadow-lg flex items-center gap-3 ${
          notification.type === 'success' ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
        }`}>
          {notification.type === 'success' ? <CheckCircle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
          {notification.message}
        </div>
      )}

      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">AI Investment Advisor</h1>
              <p className="text-sm text-gray-600">Welcome, {auth.username}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition"
          >
            <LogOut className="w-5 h-5" />
            Logout
          </button>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex gap-2 mb-6 overflow-x-auto">
          {['portfolio', 'add', 'analysis', 'rebalance'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 rounded-lg font-medium whitespace-nowrap transition ${
                activeTab === tab
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              {tab === 'portfolio' && '📊 Portfolio'}
              {tab === 'add' && '➕ Add Stock'}
              {tab === 'analysis' && '🤖 AI Analysis'}
              {tab === 'rebalance' && '⚖️ Rebalance'}
            </button>
          ))}
        </div>

        {activeTab === 'portfolio' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Your Portfolio</h2>
                <button
                  onClick={loadPortfolio}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                >
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </button>
              </div>

              {portfolio.length === 0 ? (
                <div className="text-center py-12">
                  <PieIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-600 text-lg">No stocks in portfolio yet</p>
                  <button
                    onClick={() => setActiveTab('add')}
                    className="mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                  >
                    Add Your First Stock
                  </button>
                </div>
              ) : (
                <div className="grid gap-4">
                  {portfolio.map((stock, idx) => (
                    <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <span className="text-xl font-bold text-gray-900">{stock.ticker}</span>
                            <span className="px-3 py-1 bg-blue-100 text-blue-700 text-sm font-medium rounded-full">
                              {stock.quantity} shares
                            </span>
                          </div>
                          <div className="text-sm text-gray-600">
                            Purchase Price: ${stock.purchase_price.toFixed(2)} | Total Cost: ${(stock.quantity * stock.purchase_price).toFixed(2)}
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            Added: {new Date(stock.purchase_date).toLocaleDateString()}
                          </div>
                        </div>
                        <button
                          onClick={() => removeStock(stock.ticker)}
                          className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'add' && (
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Add New Stock</h2>
            
            <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <h3 className="font-bold text-blue-900 mb-2">🌍 Global Market Support</h3>
              <p className="text-sm text-blue-800 mb-3">You can add stocks from ANY market worldwide! Just use the correct ticker format:</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-blue-700">
                <div>
                  <span className="font-bold">🇺🇸 US Stocks:</span> AAPL, MSFT, GOOGL, TSLA
                </div>
                <div>
                  <span className="font-bold">🇮🇳 India NSE:</span> RELIANCE.NS, TCS.NS, INFY.NS
                </div>
                <div>
                  <span className="font-bold">🇮🇳 India BSE:</span> RELIANCE.BO, TCS.BO, INFY.BO
                </div>
                <div>
                  <span className="font-bold">🇬🇧 UK:</span> HSBA.L, BP.L, GSK.L, AZN.L
                </div>
                <div>
                  <span className="font-bold">🇯🇵 Japan:</span> 7203.T (Toyota), 9984.T (SoftBank)
                </div>
                <div>
                  <span className="font-bold">🇭🇰 Hong Kong:</span> 0700.HK (Tencent), 0939.HK
                </div>
                <div>
                  <span className="font-bold">🇩🇪 Germany:</span> SIE.DE (Siemens), SAP
                </div>
                <div>
                  <span className="font-bold">🇫🇷 France:</span> MC.PA (LVMH), OR.PA (L'Oreal)
                </div>
              </div>
              <p className="text-xs text-blue-600 mt-3">💡 Tip: Search "Yahoo Finance [company name]" to find the correct ticker format</p>
            </div>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Stock Ticker Symbol</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newStock.ticker}
                    onChange={(e) => setNewStock({ ...newStock, ticker: e.target.value.toUpperCase() })}
                    placeholder="e.g., RELIANCE.NS, AAPL, TCS.NS"
                    className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <button
                    onClick={searchStock}
                    disabled={!newStock.ticker || loading}
                    className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition disabled:opacity-50"
                  >
                    Search
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">Examples: RELIANCE.NS (Reliance India), AAPL (Apple US), TCS.NS (TCS India)</p>
              </div>

              {searchResult && (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <h3 className="font-bold text-green-900 text-lg">{searchResult.name}</h3>
                  <p className="text-green-700 mt-1">
                    Current Price: ${searchResult.current_price.toFixed(2)} {searchResult.currency}
                  </p>
                  <p className="text-sm text-green-600 mt-1">
                    {searchResult.sector} - {searchResult.industry}
                  </p>
                </div>
              )}

              <form onSubmit={addStock} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Number of Shares</label>
                  <input
                    type="number"
                    step="0.01"
                    value={newStock.quantity}
                    onChange={(e) => setNewStock({ ...newStock, quantity: e.target.value })}
                    placeholder="e.g., 10"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Purchase Price per Share ($)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={newStock.purchase_price}
                    onChange={(e) => setNewStock({ ...newStock, purchase_price: e.target.value })}
                    placeholder="e.g., 150.00"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    required
                  />
                </div>

                {newStock.quantity && newStock.purchase_price && (
                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-blue-900 font-medium">
                      Total Investment: ${(parseFloat(newStock.quantity) * parseFloat(newStock.purchase_price)).toFixed(2)}
                    </p>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading || !newStock.ticker || !newStock.quantity || !newStock.purchase_price}
                  className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 transition disabled:opacity-50"
                >
                  {loading ? 'Adding...' : 'Add to Portfolio'}
                </button>
              </form>
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900">AI-Powered Analysis</h2>
                <button
                  onClick={runAnalysis}
                  disabled={loading || portfolio.length === 0}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition disabled:opacity-50"
                >
                  <Activity className="w-5 h-5" />
                  {loading ? 'Analyzing...' : 'Run Analysis'}
                </button>
              </div>

              {!analysis && portfolio.length === 0 && (
                <div className="text-center py-12">
                  <AlertCircle className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-600 text-lg">Add stocks to your portfolio to run analysis</p>
                </div>
              )}

              {!analysis && portfolio.length > 0 && (
                <div className="text-center py-12">
                  <Activity className="w-16 h-16 text-blue-300 mx-auto mb-4" />
                  <p className="text-gray-600 text-lg">Click "Run Analysis" to get AI-powered insights</p>
                </div>
              )}

              {analysis && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-4 text-white">
                      <div className="flex items-center gap-2 mb-2">
                        <DollarSign className="w-5 h-5" />
                        <span className="text-sm opacity-90">Total Value</span>
                      </div>
                      <p className="text-2xl font-bold">${analysis.total_value.toLocaleString()}</p>
                    </div>

                    <div className={`rounded-xl p-4 text-white ${
                      analysis.total_profit_loss >= 0
                        ? 'bg-gradient-to-br from-green-500 to-green-600'
                        : 'bg-gradient-to-br from-red-500 to-red-600'
                    }`}>
                      <div className="flex items-center gap-2 mb-2">
                        {analysis.total_profit_loss >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                        <span className="text-sm opacity-90">Total P/L</span>
                      </div>
                      <p className="text-2xl font-bold">${analysis.total_profit_loss.toLocaleString()}</p>
                    </div>

                    <div className={`rounded-xl p-4 text-white ${
                      analysis.avg_return >= 0
                        ? 'bg-gradient-to-br from-green-500 to-green-600'
                        : 'bg-gradient-to-br from-red-500 to-red-600'
                    }`}>
                      <div className="flex items-center gap-2 mb-2">
                        <Activity className="w-5 h-5" />
                        <span className="text-sm opacity-90">Avg Return</span>
                      </div>
                      <p className="text-2xl font-bold">{analysis.avg_return.toFixed(2)}%</p>
                    </div>

                    <div className={`rounded-xl p-4 text-white ${
                      analysis.diversification_score >= 70 ? 'bg-gradient-to-br from-green-500 to-green-600' :
                      analysis.diversification_score >= 50 ? 'bg-gradient-to-br from-blue-500 to-blue-600' :
                      analysis.diversification_score >= 30 ? 'bg-gradient-to-br from-yellow-500 to-yellow-600' :
                      'bg-gradient-to-br from-red-500 to-red-600'
                    }`}>
                      <div className="flex items-center gap-2 mb-2">
                        <PieIcon className="w-5 h-5" />
                        <span className="text-sm opacity-90">Diversification</span>
                      </div>
                      <p className="text-2xl font-bold">{analysis.diversification_score}/100</p>
                    </div>

                    <div className={`rounded-xl p-4 text-white ${
                      analysis.portfolio_health.includes('Excellent') ? 'bg-gradient-to-br from-green-500 to-green-600' :
                      analysis.portfolio_health.includes('Good') ? 'bg-gradient-to-br from-blue-500 to-blue-600' :
                      analysis.portfolio_health.includes('Fair') ? 'bg-gradient-to-br from-yellow-500 to-yellow-600' :
                      'bg-gradient-to-br from-red-500 to-red-600'
                    }`}>
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="w-5 h-5" />
                        <span className="text-sm opacity-90">Health</span>
                      </div>
                      <p className="text-lg font-bold">{analysis.portfolio_health}</p>
                    </div>
                  </div>

                  {/* Sector Breakdown */}
                  {analysis.sector_analysis && Object.keys(analysis.sector_analysis).length > 0 && (
                    <div className="border border-gray-200 rounded-xl p-4">
                      <h3 className="font-bold text-gray-900 mb-3">Sector Breakdown</h3>
                      <div className="space-y-2">
                        {Object.entries(analysis.sector_analysis).map(([sector, weight]) => (
                          <div key={sector} className="flex items-center justify-between">
                            <span className="text-sm text-gray-700">{sector}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                                <div 
                                  className="h-full bg-blue-600 rounded-full"
                                  style={{ width: `${weight}%` }}
                                />
                              </div>
                              <span className="text-sm font-medium text-gray-900 w-12 text-right">{weight.toFixed(1)}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                      <p className="text-xs text-gray-500 mt-3">
                        {analysis.num_stocks} stocks across {analysis.num_sectors} sectors
                      </p>
                    </div>
                  )}

                  {/* Warnings and Concerns Section */}
                  {(analysis.warnings && analysis.warnings.length > 0) && (
                    <div className="border-2 border-red-300 bg-red-50 rounded-xl p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 bg-red-600 rounded-full flex items-center justify-center">
                          <AlertCircle className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h3 className="font-bold text-red-900 text-xl">⚠️ Portfolio Concerns & Warnings</h3>
                          <p className="text-red-700 text-sm">Critical issues that need your attention</p>
                        </div>
                      </div>

                      <div className="space-y-3">
                        {analysis.warnings.map((warning, idx) => (
                          <div key={idx} className="bg-white border border-red-200 rounded-lg p-4">
                            <div className="flex items-start gap-3">
                              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
                              <div>
                                <p className="font-bold text-red-900 mb-1">{warning.title}</p>
                                <p className="text-red-700 text-sm">{warning.description}</p>
                                {warning.severity && (
                                  <span className={`inline-block mt-2 px-2 py-1 rounded text-xs font-bold ${
                                    warning.severity === 'High' ? 'bg-red-600 text-white' :
                                    warning.severity === 'Medium' ? 'bg-orange-500 text-white' :
                                    'bg-yellow-500 text-white'
                                  }`}>
                                    {warning.severity} Severity
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* AI Suggested Stocks to BUY */}
                  {analysis.suggested_stocks && analysis.suggested_stocks.length > 0 && (
                    <div className="border-2 border-green-300 bg-green-50 rounded-xl p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center">
                          <Plus className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h3 className="font-bold text-green-900 text-xl">🎯 Portfolio Diversification Recommendations</h3>
                          <p className="text-green-700 text-sm">AI-powered suggestions to strengthen your portfolio</p>
                        </div>
                      </div>

                      <div className="grid gap-4">
                        {analysis.suggested_stocks.map((stock, idx) => (
                          <div key={idx} className="bg-white border border-green-200 rounded-lg p-4 hover:shadow-lg transition">
                            {stock.ticker === 'DIVERSIFICATION_NEEDED' || stock.ticker === 'ADD_MORE_STOCKS' ? (
                              <div>
                                <div className="flex items-center gap-3 mb-3">
                                  <AlertCircle className="w-8 h-8 text-orange-600" />
                                  <div>
                                    <span className="text-xl font-bold text-orange-900">{stock.strength}</span>
                                    <p className="text-sm text-orange-700">{stock.sector}</p>
                                  </div>
                                </div>
                                <div className="space-y-2 text-sm text-gray-700">
                                  {stock.reasons.map((reason, i) => (
                                    <p key={i} className={reason.startsWith('🎯') || reason.startsWith('💡') || reason.startsWith('📊') || reason.startsWith('✓') || reason.startsWith('⚠️') ? 'font-medium' : ''}>
                                      {reason}
                                    </p>
                                  ))}
                                </div>
                                {stock.estimated_investment > 0 && (
                                  <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded">
                                    <p className="text-sm text-blue-900">
                                      <span className="font-bold">Recommended Additional Investment:</span> ${stock.estimated_investment.toLocaleString()}
                                    </p>
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="flex items-start justify-between mb-3">
                                <div className="flex-1">
                                  <div className="flex items-center gap-3 mb-2">
                                    <span className="text-2xl font-bold text-green-900">{stock.ticker}</span>
                                    <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                                      stock.strength === 'Strong Buy' ? 'bg-green-600 text-white' :
                                      stock.strength === 'Buy' ? 'bg-green-500 text-white' :
                                      'bg-green-100 text-green-700'
                                    }`}>
                                      {stock.strength}
                                    </span>
                                    <span className="px-3 py-1 bg-blue-100 text-blue-700 text-sm font-medium rounded-full">
                                      {stock.sector}
                                    </span>
                                  </div>
                                  <div className="flex items-center gap-4 text-sm mb-2">
                                    <span className="text-gray-700">Price: <span className="font-bold">${stock.current_price.toFixed(2)}</span></span>
                                    <span className={stock.returns_30d >= 0 ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
                                      30d: {stock.returns_30d >= 0 ? '+' : ''}{stock.returns_30d.toFixed(1)}%
                                    </span>
                                  </div>
                                  <div className="bg-blue-50 border border-blue-200 rounded p-3 mb-2">
                                    <p className="text-sm text-blue-900">
                                      <span className="font-bold">Suggested Investment:</span> Buy ~{stock.suggested_shares} shares 
                                      <span className="font-bold ml-2">(${stock.estimated_investment.toLocaleString()})</span>
                                    </p>
                                  </div>
                                  <div className="space-y-1">
                                    <p className="text-xs font-bold text-gray-700 mb-1">Why this stock:</p>
                                    {stock.reasons.map((reason, i) => (
                                      <div key={i} className="flex items-start gap-2 text-sm text-gray-700">
                                        <span className="text-green-600 mt-1">✓</span>
                                        <span>{reason}</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>

                      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                        <p className="text-xs text-yellow-800">
                          💡 <span className="font-bold">Tip:</span> You can add stocks from any global market (US, India NSE/BSE, UK, Japan, etc.). 
                          Use the correct ticker format (e.g., RELIANCE.NS for NSE, RELIANCE.BO for BSE).
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="border border-gray-200 rounded-xl p-4">
                      <h3 className="font-bold text-gray-900 mb-4">Portfolio Allocation</h3>
                      <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                          <Pie
                            data={analysis.analysis.map((s, i) => ({
                              name: s.ticker,
                              value: s.total_value,
                              weight: s.weight
                            }))}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={(entry) => `${entry.name} (${entry.weight.toFixed(1)}%)`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {analysis.analysis.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip formatter={(value) => `${value.toFixed(2)}`} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="border border-gray-200 rounded-xl p-4">
                      <h3 className="font-bold text-gray-900 mb-4">Performance by Stock</h3>
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={analysis.analysis}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="ticker" />
                          <YAxis />
                          <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                          <Bar dataKey="profit_loss_pct" fill="#3b82f6" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-bold text-gray-900 text-xl mb-4">🤖 Advanced AI Analysis - Your Holdings</h3>
                    <div className="grid gap-6">
                      {analysis.analysis && analysis.analysis.map((stock, idx) => (
                        <div key={idx} className={`border-2 rounded-xl p-5 hover:shadow-xl transition ${
                          stock.recommendation.action.includes('SELL') || stock.recommendation.action.includes('STRONG_SELL') 
                            ? 'bg-red-50 border-red-300' 
                            : stock.recommendation.action.includes('BUY')
                            ? 'bg-green-50 border-green-300'
                            : 'bg-white border-gray-200'
                        }`}>
                          {/* Stock Header */}
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex-1">
                              <div className="flex items-center gap-3 mb-2">
                                <span className="text-2xl font-bold text-gray-900">{stock.ticker}</span>
                                <span className={`px-4 py-1 rounded-full text-sm font-bold border-2 ${getActionColor(stock.recommendation.action)}`}>
                                  {stock.recommendation.action.replace(/_/g, ' ')}
                                </span>
                                <span className="px-3 py-1 bg-purple-100 text-purple-700 text-xs font-medium rounded-full">
                                  {stock.recommendation.confidence} Confidence
                                </span>
                                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                                  {stock.sector}
                                </span>
                              </div>
                              
                              {/* Price & Performance */}
                              <div className="flex items-center gap-4 text-sm mb-2">
                                <span className="font-semibold text-gray-900">
                                  Price: ${stock.current_price.toFixed(2)}
                                </span>
                                <span className={stock.profit_loss_pct >= 0 ? 'text-green-600 font-bold' : 'text-red-600 font-bold'}>
                                  {stock.profit_loss_pct >= 0 ? '+' : ''}{stock.profit_loss_pct.toFixed(2)}% 
                                  ({stock.profit_loss_pct >= 0 ? '+' : ''}${stock.profit_loss.toFixed(2)})
                                </span>
                                <span className="text-gray-600">
                                  Weight: {stock.weight.toFixed(1)}%
                                </span>
                              </div>

                              {/* AI Summary */}
                              <div className={`border rounded-lg p-3 mb-3 ${
                                stock.recommendation.action.includes('SELL') || stock.recommendation.action.includes('STRONG_SELL')
                                  ? 'bg-red-100 border-red-300'
                                  : stock.recommendation.action.includes('BUY')
                                  ? 'bg-green-100 border-green-300'
                                  : 'bg-blue-50 border-blue-200'
                              }`}>
                                <p className={`text-sm font-semibold ${
                                  stock.recommendation.action.includes('SELL') || stock.recommendation.action.includes('STRONG_SELL')
                                    ? 'text-red-900'
                                    : stock.recommendation.action.includes('BUY')
                                    ? 'text-green-900'
                                    : 'text-blue-900'
                                }`}>
                                  🤖 AI Analysis: {stock.recommendation.summary}
                                </p>
                              </div>
                            </div>
                          </div>

                          {/* Future Prediction */}
                          {stock.prediction && (
                            <div className={`border-2 rounded-lg p-4 mb-3 ${
                              stock.prediction.change_pct >= 0 
                                ? 'bg-gradient-to-r from-green-50 to-blue-50 border-green-200'
                                : 'bg-gradient-to-r from-red-50 to-orange-50 border-red-200'
                            }`}>
                              <div className="flex items-center justify-between mb-2">
                                <h4 className={`font-bold flex items-center gap-2 ${
                                  stock.prediction.change_pct >= 0 ? 'text-green-900' : 'text-red-900'
                                }`}>
                                  {stock.prediction.change_pct >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                                  30-Day AI Price Prediction
                                </h4>
                                <span className={`px-3 py-1 text-white text-xs font-bold rounded-full ${
                                  stock.prediction.confidence >= 80 ? 'bg-green-600' :
                                  stock.prediction.confidence >= 60 ? 'bg-blue-600' :
                                  'bg-orange-600'
                                }`}>
                                  {stock.prediction.confidence}% Confidence
                                </span>
                              </div>
                              <div className="grid grid-cols-3 gap-4 text-center">
                                <div>
                                  <p className="text-xs text-gray-600 mb-1">Current</p>
                                  <p className="text-lg font-bold text-gray-900">${stock.prediction.current_price}</p>
                                </div>
                                <div>
                                  <p className="text-xs text-gray-600 mb-1">Predicted (30d)</p>
                                  <p className="text-lg font-bold text-gray-900">${stock.prediction.predicted_price}</p>
                                </div>
                                <div>
                                  <p className="text-xs text-gray-600 mb-1">Expected Change</p>
                                  <p className={`text-lg font-bold ${stock.prediction.change_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                    {stock.prediction.change_pct >= 0 ? '+' : ''}{stock.prediction.change_pct}%
                                  </p>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Technical Indicators Grid */}
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                            {/* RSI */}
                            <div className={`border rounded-lg p-3 ${
                              stock.rsi > 70 ? 'bg-red-50 border-red-200' :
                              stock.rsi < 30 ? 'bg-green-50 border-green-200' :
                              'bg-white border-gray-200'
                            }`}>
                              <p className="text-xs text-gray-600 mb-1">RSI</p>
                              <p className={`text-lg font-bold ${
                                stock.rsi > 70 ? 'text-red-600' :
                                stock.rsi < 30 ? 'text-green-600' :
                                'text-gray-900'
                              }`}>
                                {stock.rsi.toFixed(1)}
                              </p>
                              <p className="text-xs text-gray-500">
                                {stock.rsi > 70 ? '🚨 Overbought' :
                                 stock.rsi < 30 ? '📈 Oversold' :
                                 '⚖️ Neutral'}
                              </p>
                            </div>

                            {/* MACD */}
                            <div className={`border rounded-lg p-3 ${
                              stock.macd > stock.macd_signal ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
                            }`}>
                              <p className="text-xs text-gray-600 mb-1">MACD</p>
                              <p className={`text-lg font-bold ${
                                stock.macd > stock.macd_signal ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {stock.macd_histogram >= 0 ? '+' : ''}{stock.macd_histogram.toFixed(3)}
                              </p>
                              <p className="text-xs text-gray-500">
                                {stock.macd > stock.macd_signal ? '📈 Bullish' : '📉 Bearish'}
                              </p>
                            </div>

                            {/* Volatility */}
                            <div className={`border rounded-lg p-3 ${
                              stock.volatility > 0.4 ? 'bg-red-50 border-red-200' :
                              stock.volatility > 0.25 ? 'bg-yellow-50 border-yellow-200' :
                              'bg-green-50 border-green-200'
                            }`}>
                              <p className="text-xs text-gray-600 mb-1">Volatility</p>
                              <p className={`text-lg font-bold ${
                                stock.volatility > 0.4 ? 'text-red-600' :
                                stock.volatility > 0.25 ? 'text-yellow-600' :
                                'text-green-600'
                              }`}>
                                {(stock.volatility * 100).toFixed(1)}%
                              </p>
                              <p className="text-xs text-gray-500">
                                {stock.volatility > 0.4 ? '⚠️ High' :
                                 stock.volatility > 0.25 ? '📊 Moderate' :
                                 '✅ Low'}
                              </p>
                            </div>

                            {/* Sharpe Ratio */}
                            <div className={`border rounded-lg p-3 ${
                              stock.sharpe_ratio > 1 ? 'bg-green-50 border-green-200' :
                              stock.sharpe_ratio > 0 ? 'bg-blue-50 border-blue-200' :
                              'bg-red-50 border-red-200'
                            }`}>
                              <p className="text-xs text-gray-600 mb-1">Sharpe Ratio</p>
                              <p className={`text-lg font-bold ${
                                stock.sharpe_ratio > 1 ? 'text-green-600' :
                                stock.sharpe_ratio > 0 ? 'text-blue-600' :
                                'text-red-600'
                              }`}>
                                {stock.sharpe_ratio.toFixed(2)}
                              </p>
                              <p className="text-xs text-gray-500">
                                {stock.sharpe_ratio > 1 ? '⭐ Excellent' :
                                 stock.sharpe_ratio > 0 ? '👍 Good' :
                                 '👎 Poor'}
                              </p>
                            </div>
                          </div>

                          {/* Risk Analysis */}
                          {stock.risk_analysis && (
                            <div className={`border-2 rounded-lg p-3 mb-3 ${
                              stock.risk_analysis.level === 'Very High' ? 'bg-red-100 border-red-300' :
                              stock.risk_analysis.level === 'High' ? 'bg-red-50 border-red-200' :
                              stock.risk_analysis.level === 'Moderate' ? 'bg-yellow-50 border-yellow-300' :
                              stock.risk_analysis.level === 'Low' ? 'bg-green-50 border-green-300' :
                              'bg-blue-50 border-blue-300'
                            }`}>
                              <div className="flex items-center justify-between mb-2">
                                <h4 className="font-bold text-gray-900">Risk Analysis</h4>
                                <span className={`px-3 py-1 rounded-full text-xs font-bold ${getRiskColor(stock.risk_analysis.level)}`}>
                                  {stock.risk_analysis.level} Risk ({stock.risk_analysis.score}/100)
                                </span>
                              </div>
                              <div className="text-xs text-gray-700 space-y-1">
                                {stock.risk_analysis.factors.map((factor, i) => (
                                  <div key={i} className={`flex items-start gap-2 ${
                                    factor.includes('High') || factor.includes('Warning') || factor.includes('Caution') 
                                      ? 'text-red-600 font-medium' 
                                      : ''
                                  }`}>
                                    <span>•</span>
                                    <span>{factor}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Performance Timeline */}
                          <div className="grid grid-cols-4 gap-2 mb-3">
                            {[
                              { label: '7 Days', value: stock.returns_7d },
                              { label: '30 Days', value: stock.returns_30d },
                              { label: '90 Days', value: stock.returns_90d },
                              { label: '1 Year', value: stock.returns_1y }
                            ].map((period, i) => (
                              <div key={i} className={`border rounded p-2 text-center ${getPerformanceColor(period.value)}`}>
                                <p className="text-xs text-gray-600">{period.label}</p>
                                <p className={`text-sm font-bold ${period.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                  {period.value >= 0 ? '+' : ''}{period.value.toFixed(1)}%
                                </p>
                              </div>
                            ))}
                          </div>

                          {/* AI Reasons */}
                          <div className="space-y-1">
                            <p className="text-xs font-bold text-gray-700 mb-2">📊 Technical Analysis Details:</p>
                            {stock.recommendation.reasons.map((reason, i) => (
                              <div key={i} className={`flex items-start gap-2 text-sm p-2 rounded ${
                                reason.includes('Warning') || reason.includes('Caution') || reason.includes('High risk') || reason.includes('Overvalued')
                                  ? 'bg-red-50 text-red-700 border border-red-200'
                                  : reason.includes('Positive') || reason.includes('Strong') || reason.includes('Undervalued')
                                  ? 'bg-green-50 text-green-700 border border-green-200'
                                  : 'bg-gray-50 text-gray-700 border border-gray-200'
                              }`}>
                                <span className={
                                  reason.includes('Warning') || reason.includes('Caution') || reason.includes('High risk') ? 'text-red-600' :
                                  reason.includes('Positive') || reason.includes('Strong') ? 'text-green-600' :
                                  'text-blue-600'
                                }>
                                  {reason.includes('Warning') || reason.includes('Caution') ? '⚠️' :
                                   reason.includes('Positive') ? '✅' : '📊'}
                                </span>
                                <span>{reason}</span>
                              </div>
                            ))}
                          </div>

                          {/* Additional Info */}
                          <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-600 flex items-center gap-4">
                            <span>Volume: {stock.volume_trend}</span>
                            {stock.pe_ratio && (
                              <span className={stock.pe_ratio > 25 ? 'text-orange-600 font-medium' : 'text-gray-600'}>
                                P/E: {stock.pe_ratio} {stock.pe_ratio > 25 ? '(High)' : ''}
                              </span>
                            )}
                            <span className={stock.beta > 1.2 ? 'text-red-600 font-medium' : stock.beta < 0.8 ? 'text-green-600 font-medium' : 'text-gray-600'}>
                              Beta: {stock.beta} {stock.beta > 1.2 ? '(Volatile)' : stock.beta < 0.8 ? '(Stable)' : ''}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Portfolio Action Items */}
                  {analysis.suggested_actions && analysis.suggested_actions.length > 0 && (
                    <div className="bg-gradient-to-r from-orange-50 to-red-50 border-2 border-orange-300 rounded-xl p-6">
                      <h3 className="font-bold text-orange-900 text-xl mb-4 flex items-center gap-2">
                        <AlertCircle className="w-6 h-6" />
                        🎯 Priority Action Items
                      </h3>
                      <div className="space-y-3">
                        {analysis.suggested_actions.map((action, idx) => (
                          <div key={idx} className={`bg-white border-l-4 rounded-lg p-4 ${
                            action.priority === 'High' ? 'border-red-500' :
                            action.priority === 'Medium' ? 'border-yellow-500' :
                            'border-blue-500'
                          }`}>
                            <div className="flex items-start gap-3">
                              <span className={`px-2 py-1 rounded text-xs font-bold ${
                                action.priority === 'High' ? 'bg-red-100 text-red-700' :
                                action.priority === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                                'bg-blue-100 text-blue-700'
                              }`}>
                                {action.priority}
                              </span>
                              <div className="flex-1">
                                <p className="font-bold text-gray-900 mb-1">{action.action}</p>
                                <p className="text-sm text-gray-600">{action.reason}</p>
                                <span className="text-xs text-gray-500 mt-1 inline-block">Type: {action.type}</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'rebalance' && (
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Portfolio Rebalancing</h2>
              <button
                onClick={runRebalancing}
                disabled={loading || portfolio.length < 2}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition disabled:opacity-50"
              >
                <RefreshCw className="w-5 h-5" />
                {loading ? 'Analyzing...' : 'Analyze Rebalancing'}
              </button>
            </div>

            {!rebalancing && portfolio.length < 2 && (
              <div className="text-center py-12">
                <AlertCircle className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-600 text-lg">Add at least 2 stocks to analyze rebalancing</p>
              </div>
            )}

            {!rebalancing && portfolio.length >= 2 && (
              <div className="text-center py-12">
                <RefreshCw className="w-16 h-16 text-purple-300 mx-auto mb-4" />
                <p className="text-gray-600 text-lg">Click "Analyze Rebalancing" to optimize your portfolio</p>
              </div>
            )}

            {rebalancing && Object.keys(rebalancing).length > 0 && (
              <div className="space-y-6">
                {/* Summary Stats */}
                {rebalancing.total_portfolio_value && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-4 text-white">
                      <p className="text-sm opacity-90 mb-1">Total Portfolio Value</p>
                      <p className="text-2xl font-bold">${rebalancing.total_portfolio_value.toLocaleString()}</p>
                    </div>
                    <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-4 text-white">
                      <p className="text-sm opacity-90 mb-1">Number of Stocks</p>
                      <p className="text-2xl font-bold">{rebalancing.num_stocks || 0}</p>
                    </div>
                    <div className="bg-gradient-to-br from-pink-500 to-pink-600 rounded-xl p-4 text-white">
                      <p className="text-sm opacity-90 mb-1">Target Weight per Stock</p>
                      <p className="text-2xl font-bold">{rebalancing.target_weight?.toFixed(1) || 0}%</p>
                    </div>
                  </div>
                )}

                {/* Rebalancing Status */}
                {typeof rebalancing.rebalancing_needed === 'boolean' && (
                  <>
                    {rebalancing.rebalancing_needed ? (
                      <div className="p-4 bg-yellow-50 border-2 border-yellow-300 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertCircle className="w-5 h-5 text-yellow-600" />
                          <span className="font-bold text-yellow-900">⚖️ Rebalancing Recommended</span>
                        </div>
                        <p className="text-yellow-800">Your portfolio allocation is imbalanced. Consider the following actions to achieve equal weighting:</p>
                      </div>
                    ) : (
                      <div className="p-4 bg-green-50 border-2 border-green-300 rounded-lg">
                        <div className="flex items-center gap-2">
                          <CheckCircle className="w-5 h-5 text-green-600" />
                          <span className="font-bold text-green-900">✅ Portfolio is Well Balanced</span>
                        </div>
                        <p className="text-green-800 mt-1">Your current allocation is optimal. No rebalancing needed at this time.</p>
                      </div>
                    )}
                  </>
                )}

                {/* Show message if not enough stocks */}
                {rebalancing.message && (
                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-blue-800">{rebalancing.message}</p>
                  </div>
                )}

                {/* Rebalancing Actions - FIXED: Safely access nested rebalancing object */}
                {rebalancing.rebalancing && rebalancing.rebalancing.rebalancing_needed && rebalancing.rebalancing.actions && (
                  <div>
                    <h3 className="font-bold text-gray-900 text-lg mb-4">🎯 Recommended Rebalancing Actions</h3>
                    <div className="space-y-3">
                      {rebalancing.rebalancing.actions.map((action, idx) => (
                        <div key={idx} className="bg-white border-2 border-gray-200 rounded-lg p-4 hover:shadow-md transition">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex-1">
                              <div className="flex items-center gap-3 mb-2">
                                <span className="text-xl font-bold text-gray-900">{action.ticker}</span>
                                <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                                  action.difference > 0 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                                }`}>
                                  {action.difference > 0 ? 'Over-weighted' : 'Under-weighted'}
                                </span>
                              </div>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-3">
                                <div>
                                  <p className="text-gray-600 text-xs">Current Weight</p>
                                  <p className="font-bold text-gray-900">{action.current_weight}%</p>
                                </div>
                                <div>
                                  <p className="text-gray-600 text-xs">Target Weight</p>
                                  <p className="font-bold text-gray-900">{action.target_weight}%</p>
                                </div>
                                <div>
                                  <p className="text-gray-600 text-xs">Difference</p>
                                  <p className={`font-bold ${action.difference > 0 ? 'text-red-600' : 'text-green-600'}`}>
                                    {action.difference > 0 ? '+' : ''}{action.difference.toFixed(1)}%
                                  </p>
                                </div>
                                <div>
                                  <p className="text-gray-600 text-xs">Shares to Trade</p>
                                  <p className="font-bold text-gray-900">{Math.abs(action.shares_to_trade)}</p>
                                </div>
                              </div>
                              <div className="bg-blue-50 border border-blue-200 rounded p-3">
                                <p className="text-sm font-medium text-blue-900">
                                  💡 Action: {action.action}
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Current vs Target Allocation Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Current Allocation */}
                  <div className="border-2 border-gray-200 rounded-xl p-4">
                    <h3 className="font-bold text-gray-900 mb-4">📊 Current Allocation</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={rebalancing.current_allocation}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={(entry) => `${entry.ticker}: ${entry.weight.toFixed(1)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="weight"
                        >
                          {rebalancing.current_allocation.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="mt-4 space-y-2">
                      {rebalancing.current_allocation.map((stock, idx) => (
                        <div key={idx} className="flex items-center justify-between text-sm">
                          <div className="flex items-center gap-2">
                            <div 
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                            />
                            <span className="font-medium">{stock.ticker}</span>
                          </div>
                          <div className="text-right">
                            <span className="font-bold">{stock.weight.toFixed(1)}%</span>
                            <span className="text-gray-500 ml-2">(${stock.value?.toLocaleString()})</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Target Equal Weight */}
                  <div className="border-2 border-green-200 bg-green-50 rounded-xl p-4">
                    <h3 className="font-bold text-green-900 mb-4">🎯 Target Equal Weight</h3>
                    <div className="flex items-center justify-center h-64">
                      <div className="text-center">
                        <p className="text-6xl font-bold text-green-600 mb-3">
                          {rebalancing.target_weight?.toFixed(1)}%
                        </p>
                        <p className="text-lg text-green-800 font-medium">per stock</p>
                        <p className="text-sm text-green-600 mt-4">Equal weight distribution</p>
                        <p className="text-xs text-green-600 mt-2">
                          ({rebalancing.num_stocks} stocks)
                        </p>
                      </div>
                    </div>
                    <div className="mt-4 p-3 bg-white rounded border border-green-300">
                      <p className="text-xs text-green-800">
                        💡 <span className="font-bold">Goal:</span> Each stock should represent approximately {rebalancing.target_weight?.toFixed(1)}% of your total portfolio value for optimal diversification.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Rebalancing Tips */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-bold text-blue-900 mb-3">📚 Rebalancing Best Practices</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-blue-800">
                    <div className="flex items-start gap-2">
                      <span className="text-blue-600 mt-1">•</span>
                      <span><strong>Frequency:</strong> Rebalance quarterly or when weights differ by &gt;10%</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-blue-600 mt-1">•</span>
                      <span><strong>Tax Impact:</strong> Consider capital gains taxes before selling</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-blue-600 mt-1">•</span>
                      <span><strong>Transaction Costs:</strong> Factor in brokerage fees</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-blue-600 mt-1">•</span>
                      <span><strong>Gradual Approach:</strong> Rebalance over time rather than all at once</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}