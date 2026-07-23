document.addEventListener('DOMContentLoaded', () => {
  // 1. Highlight active nav items
  const path = window.location.pathname;
  document.querySelectorAll('.nav-item[data-page]').forEach(item => {
    const pageAttr = item.getAttribute('data-page');
    if (path.includes(pageAttr)) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // 2. Initialize Dark Mode from localStorage or system preference
  const savedTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
    document.documentElement.classList.add('dark-mode');
    updateThemeIcon(true);
  } else {
    document.documentElement.classList.remove('dark-mode');
    updateThemeIcon(false);
  }

  // 3. Setup Theme Toggle event listener
  const themeToggleBtn = document.getElementById('themeToggle');
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const isDark = document.documentElement.classList.toggle('dark-mode');
      localStorage.setItem('theme', isDark ? 'dark' : 'light');
      updateThemeIcon(isDark);
      showNotification(`Switched to ${isDark ? 'Dark' : 'Light'} Mode`, 'info');
    });
  }

  // 4. Setup Mobile Sidebar toggles
  const sidebar = document.getElementById('appSidebar');
  const menuBtn = document.getElementById('mobileMenuToggle');
  if (menuBtn && sidebar) {
    menuBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      sidebar.classList.toggle('open');
    });

    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && e.target !== menuBtn) {
        sidebar.classList.remove('open');
      }
    });
  }
});

function updateThemeIcon(isDark) {
  const icon = document.querySelector('#themeToggle i');
  if (icon) {
    icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
  }
}

// Global Notification/Toast system
function showNotification(message, type = 'info', duration = 3500) {
  // Check if container exists, else create
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'notification-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `notification ${type}`;
  
  const icons = {
    'success': 'check-circle',
    'danger': 'exclamation-circle',
    'warning': 'exclamation-triangle',
    'info': 'info-circle'
  };
  
  const iconName = icons[type] || 'info-circle';
  toast.innerHTML = `
    <i class="fas fa-${iconName}"></i>
    <span style="flex:1">${message}</span>
    <i class="fas fa-times" style="font-size:12px; cursor:pointer; opacity:0.6" onclick="this.parentElement.remove()"></i>
  `;
  
  container.appendChild(toast);
  
  // Slide out and remove
  setTimeout(() => {
    toast.style.transform = 'translateX(120%)';
    toast.style.transition = 'transform 0.3s ease-in-out';
    setTimeout(() => {
      toast.remove();
    }, 300);
  }, duration);
}

// Map old function name to new one to prevent failures in existing inline html JS
window.showNotification = showNotification;
window.showNotificationDanger = (msg) => showNotification(msg, 'danger');
window.showNotificationSuccess = (msg) => showNotification(msg, 'success');
