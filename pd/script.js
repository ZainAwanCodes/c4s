// ──────────────────────────────────────────────
//  Linear Regression Model (client-side replica)
//  Coefficients from sklearn trained model:
//    StudyHours  →  1.60421778
//    Attendance  → -0.06310434
//    Assignments →  0.85829485
//    Intercept   → 10.59498124
//  R² Score     :  0.9988
// ──────────────────────────────────────────────

const MODEL = {
  coef: {
    studyHours:  1.60421778,
    attendance: -0.06310434,
    assignments: 0.85829485
  },
  intercept: 10.59498124,
  r2: 0.9988
};

const DATASET = [
  { hours: 2, attendance: 60, assignments: 45, marks: 48 },
  { hours: 5, attendance: 85, assignments: 75, marks: 78 },
  { hours: 1, attendance: 50, assignments: 35, marks: 40 },
  { hours: 6, attendance: 92, assignments: 88, marks: 90 },
  { hours: 3, attendance: 70, assignments: 60, marks: 62 },
  { hours: 7, attendance: 95, assignments: 92, marks: 96 },
  { hours: 4, attendance: 75, assignments: 68, marks: 70 },
  { hours: 8, attendance: 98, assignments: 95, marks: 98 }
];

// ─── Predict function ───
function predict(hours, attendance, assignments) {
  const val = MODEL.coef.studyHours  * hours
            + MODEL.coef.attendance  * attendance
            + MODEL.coef.assignments * assignments
            + MODEL.intercept;
  return Math.min(100, Math.max(0, val));
}

// ─── Grade helper ───
function getGrade(score) {
  if (score >= 90) return { letter: 'A+', cls: 'grade-a', msg: '🌟 Outstanding! Top-tier performance predicted.' };
  if (score >= 80) return { letter: 'A',  cls: 'grade-a', msg: '✨ Excellent result! Keep up the great work.' };
  if (score >= 70) return { letter: 'B+', cls: 'grade-b', msg: '👍 Very good performance. Minor improvements ahead.' };
  if (score >= 60) return { letter: 'B',  cls: 'grade-b', msg: '📘 Good standing. Consistent effort pays off.' };
  if (score >= 50) return { letter: 'C',  cls: 'grade-c', msg: '⚡ Average range. More study hours could help.' };
  return                  { letter: 'D',  cls: 'grade-d', msg: '📣 Below average. Attendance & assignments need focus.' };
}

// ─── DOM refs ───
const sliders = {
  studyHours:  document.getElementById('slider-hours'),
  attendance:  document.getElementById('slider-attendance'),
  assignments: document.getElementById('slider-assignments')
};
const numbers = {
  studyHours:  document.getElementById('num-hours'),
  attendance:  document.getElementById('num-attendance'),
  assignments: document.getElementById('num-assignments')
};
const badges = {
  studyHours:  document.getElementById('badge-hours'),
  attendance:  document.getElementById('badge-attendance'),
  assignments: document.getElementById('badge-assignments')
};

const btnPredict    = document.getElementById('btn-predict');
const resultScore   = document.getElementById('result-score');
const resultGrade   = document.getElementById('result-grade');
const resultMessage = document.getElementById('result-message');
const ringProgress  = document.getElementById('ring-progress');
const ringGradStart = document.getElementById('ring-grad-start');
const ringGradEnd   = document.getElementById('ring-grad-end');

// ─── Sync slider ↔ number input ───
function syncInputs(key) {
  const sl = sliders[key];
  const nm = numbers[key];
  const bd = badges[key];

  const val = parseFloat(sl.value);
  nm.value  = val;
  bd.textContent = val + (key === 'studyHours' ? ' hrs' : key === 'attendance' ? '%' : ' pts');

  // Update fill
  const min = parseFloat(sl.min), max = parseFloat(sl.max);
  const pct = ((val - min) / (max - min)) * 100;
  sl.style.background =
    `linear-gradient(90deg, #00f5ff ${pct}%, rgba(255,255,255,0.06) ${pct}%)`;
}

function syncFromNumber(key) {
  const nm = numbers[key];
  const sl = sliders[key];
  let val = parseFloat(nm.value) || 0;
  val = Math.min(parseFloat(sl.max), Math.max(parseFloat(sl.min), val));
  nm.value  = val;
  sl.value  = val;
  syncInputs(key);
}

// ─── Spin buttons ───
document.querySelectorAll('.btn-spin').forEach(btn => {
  btn.addEventListener('click', () => {
    const key  = btn.dataset.key;
    const dir  = parseInt(btn.dataset.dir);
    const nm   = numbers[key];
    const sl   = sliders[key];
    let val = parseFloat(nm.value) + dir;
    val = Math.min(parseFloat(sl.max), Math.max(parseFloat(sl.min), val));
    nm.value  = val;
    sl.value  = val;
    syncInputs(key);
  });
});

// ─── Slider & number events ───
Object.keys(sliders).forEach(key => {
  sliders[key].addEventListener('input', () => syncInputs(key));
  numbers[key].addEventListener('input', () => syncFromNumber(key));
  numbers[key].addEventListener('change', () => syncFromNumber(key));
  syncInputs(key);  // init
});

// ─── Ring progress ───
const RING_CIRCUMFERENCE = 2 * Math.PI * 85; // r=85

function updateRing(pct) {
  const offset = RING_CIRCUMFERENCE * (1 - pct / 100);
  ringProgress.style.strokeDashoffset = offset;

  // Color shift based on score
  if (pct >= 80) {
    ringGradStart.setAttribute('stop-color', '#00f5ff');
    ringGradEnd.setAttribute('stop-color', '#00e676');
  } else if (pct >= 60) {
    ringGradStart.setAttribute('stop-color', '#00f5ff');
    ringGradEnd.setAttribute('stop-color', '#a259ff');
  } else if (pct >= 40) {
    ringGradStart.setAttribute('stop-color', '#a259ff');
    ringGradEnd.setAttribute('stop-color', '#ffd700');
  } else {
    ringGradStart.setAttribute('stop-color', '#ff6b9d');
    ringGradEnd.setAttribute('stop-color', '#ffd700');
  }
}

// ─── Animated counter ───
function animateCount(from, to, duration, cb) {
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    const cur = from + (to - from) * ease;
    cb(cur);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ─── Predict ───
let currentScore = 0;

function runPrediction() {
  const hours       = parseFloat(sliders.studyHours.value);
  const attendance  = parseFloat(sliders.attendance.value);
  const assignments = parseFloat(sliders.assignments.value);

  // Animate button
  btnPredict.classList.add('loading');

  setTimeout(() => {
    const score = predict(hours, attendance, assignments);
    const grade = getGrade(score);

    btnPredict.classList.remove('loading');

    // Animate score counter
    resultScore.classList.add('animating');
    animateCount(currentScore, score, 1200, (val) => {
      resultScore.textContent = val.toFixed(1);
      updateRing(val);
    });
    setTimeout(() => resultScore.classList.remove('animating'), 1300);

    // Grade & message
    resultGrade.textContent = grade.letter;
    resultGrade.className   = 'result-grade ' + grade.cls;
    resultMessage.textContent = grade.msg;

    // Ripple
    addRipple();

    currentScore = score;
  }, 700);
}

btnPredict.addEventListener('click', runPrediction);

// ─── Ripple effect ───
function addRipple() {
  const ring = document.querySelector('.result-ring');
  const ripple = document.createElement('div');
  ripple.className = 'ripple-ring';
  ring.appendChild(ripple);
  ripple.addEventListener('animationend', () => ripple.remove());
}

// ─── Build dataset table ───
function buildTable() {
  const tbody = document.getElementById('dataset-tbody');
  tbody.innerHTML = DATASET.map((row, i) => `
    <tr style="animation: fadeSlideUp 0.4s ease ${0.05 * i}s both;">
      <td>${row.hours}</td>
      <td>${row.attendance}%</td>
      <td>${row.assignments}</td>
      <td class="td-marks">${row.marks}</td>
    </tr>
  `).join('');
}

// ─── Spawn floating particles ───
function spawnParticles() {
  const colors = ['#00f5ff', '#a259ff', '#ff6b9d', '#00e676'];
  for (let i = 0; i < 20; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left = Math.random() * 100 + 'vw';
    p.style.width = p.style.height = (Math.random() * 3 + 1) + 'px';
    p.style.background = colors[Math.floor(Math.random() * colors.length)];
    p.style.animationDuration = (Math.random() * 15 + 10) + 's';
    p.style.animationDelay = (Math.random() * 15) + 's';
    document.querySelector('.bg-canvas').appendChild(p);
  }
}

// ─── Init ───
document.addEventListener('DOMContentLoaded', () => {
  buildTable();
  spawnParticles();

  // Run initial prediction
  setTimeout(() => {
    runPrediction();
  }, 900);
});
