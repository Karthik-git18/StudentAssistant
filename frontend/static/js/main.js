document.addEventListener('DOMContentLoaded', () => {
  // Highlight active nav item based on current page
  const path = window.location.pathname;
  const pageMap = {
    '/home': 'home',
    '/learning': 'learning',
    '/chat': 'chat',
    '/planner': 'planner',
    '/profile': 'profile'
  };
  
  const page = pageMap[path];
  if (page) {
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
      if (item.getAttribute('data-page') === page) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });
  }

  // Sidebar toggle on mobile
  const sidebar = document.querySelector('.sidebar');
  if (window.innerWidth <= 768) {
    // Auto-hide on mobile
  }
});

// Global notification system
function showNotification(message, type = 'info', duration = 3000) {
  const n = document.createElement('div');
  n.className = `notification ${type}`;
  const icons = {
    'success': 'check-circle',
    'error': 'exclamation-circle',
    'info': 'info-circle'
  };
  n.innerHTML = `<i class="fas fa-${icons[type]}"></i><span>${message}</span>`;
  document.body.appendChild(n);
  
  setTimeout(() => {
    n.style.animation = 'slideOutRight 0.3s ease';
    setTimeout(() => n.remove(), 300);
  }, duration);
}

// Add slideOutRight animation
const style = document.createElement('style');
style.textContent = `
  @keyframes slideOutRight {
    from { opacity: 1; transform: translateX(0); }
    to { opacity: 0; transform: translateX(100px); }
  }
`;
document.head.appendChild(style);

document.addEventListener('DOMContentLoaded', function(){
  // highlight active nav
  const path = window.location.pathname || '/home';
  document.querySelectorAll('.bottom-nav .nav-item').forEach(a=>{
    if(a.getAttribute('href')===path) a.classList.add('active');
  });
});
