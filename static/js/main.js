/**
 * SemanticReads Frontend Script
 * Handles mobile navbar toggle, cover image fallback, and recommendation loading states.
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Mobile Navigation Menu Toggle
  const navToggle = document.querySelector('.nav-toggle');
  const navLinks = document.querySelector('.nav-links');

  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      navLinks.classList.toggle('show');
      const isExpanded = navLinks.classList.contains('show');
      navToggle.setAttribute('aria-expanded', isExpanded);
    });
  }

  // 2. Recommendation Form Submission Loading State
  const recForm = document.getElementById('recommendation-form');
  const submitBtn = document.getElementById('btn-generate-recs');

  if (recForm && submitBtn) {
    recForm.addEventListener('submit', (e) => {
      const originalText = submitBtn.innerHTML;
      submitBtn.disabled = true;
      submitBtn.style.opacity = '0.75';
      submitBtn.style.cursor = 'not-allowed';
      submitBtn.innerHTML = `
        <span class="loading-spinner"></span>
        <span>Generating personalized recommendations...</span>
      `;
    });
  }

  // 3. User Quick-Select Helper on Recommendations Page
  const userSelect = document.getElementById('user-quick-select');
  const userIdInput = document.getElementById('user_id_input');

  if (userSelect && userIdInput) {
    userSelect.addEventListener('change', (e) => {
      if (e.target.value) {
        userIdInput.value = e.target.value;
      }
    });
  }
});

/**
 * Global fallback handler for broken book cover images.
 */
function handleImageError(imgElement) {
  if (!imgElement.dataset.failed) {
    imgElement.dataset.failed = "true";
    imgElement.src = "/static/images/placeholder-book.svg";
    imgElement.alt = "Cover image unavailable";
  }
}
