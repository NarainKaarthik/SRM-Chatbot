// ------------------------------
// DOM Elements
// ------------------------------
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const timetable = document.getElementById("exam-timetable");

let isTyping = false;
let typingTimeout;
let controller;

// ------------------------------
// Mouse Parallax Effect
// ------------------------------
document.addEventListener("mousemove", (e) => {
  const mouseEffect = document.querySelector(".mouse-effect");
  mouseEffect.style.background = `radial-gradient(circle at ${e.clientX}px ${e.clientY}px, rgba(113, 15, 251, 0.34), transparent 25%)`;
});

// ------------------------------
// Terms & Conditions Modal
// ------------------------------
window.onload = () => {
  const modal = document.getElementById("tcModal");
  const acceptBtn = document.getElementById("acceptBtn");
  document.body.classList.add("modal-open");
  modal.style.display = "flex";

  acceptBtn.onclick = () => {
    modal.style.display = "none";
    document.body.classList.remove("modal-open");
  };

  modal.onclick = (e) => {
    if (e.target === modal) e.stopPropagation();
  };
};

// ------------------------------
// Render Timetable
// ------------------------------

function renderTimetable(timetable) {
    const tbody = document.querySelector("#timetable-table tbody");
    tbody.innerHTML = ""; // clear old rows

    timetable.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${row.subject}</td>
            <td>${row.code}</td>
            <td>${row.department}</td>
            <td>${row.year}</td>
            <td>${row.date}</td>
            <td>${row.time}</td>
        `;
        tbody.appendChild(tr);
    });

    const tableDiv = document.getElementById("exam-timetable");
    tableDiv.style.display = "block";            // show table
    tableDiv.style.opacity = 0;                  // start invisible
    setTimeout(() => tableDiv.style.opacity = 1, 50);  // fade in
    tableDiv.scrollIntoView({ behavior: "smooth" });
}

// ------------------------------
// Feedback Buttons
// ------------------------------
function addFeedbackButtons(msgDiv, assistantMessage) {
  const feedbackDiv = document.createElement("div");
  feedbackDiv.classList.add("feedback");
  feedbackDiv.innerHTML = `
    <span class="thumb" data-feedback="up">👍</span>
    <span class="thumb" data-feedback="down">👎</span>
  `;
  msgDiv.appendChild(feedbackDiv);

  feedbackDiv.querySelectorAll(".thumb").forEach(thumb => {
    thumb.addEventListener("click", () => {
      const feedback = thumb.dataset.feedback;
      feedbackDiv.innerHTML = `<span class="feedback-thanks">Thanks! ${feedback === "up" ? "😊" : "😔"}</span>`;

      fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: assistantMessage, feedback })
      });
    });
  });
}

// ------------------------------
// Chat Functions
// ------------------------------
function appendMessage(message, sender) {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", sender === "assistant" ? "assistant-message" : "user-message");
  msgDiv.innerHTML = `<p>${message}</p>`;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  if (sender === "assistant") addFeedbackButtons(msgDiv, message);
}

function typeEffect(text) {
  return new Promise(resolve => {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", "assistant-message");
    chatBox.appendChild(msgDiv);

    let i = 0;
    const speed = 20;
    isTyping = true;

    function type() {
      if (i < text.length && isTyping) {
        msgDiv.textContent = text.substring(0, i + 1);
        i++;
        chatBox.scrollTop = chatBox.scrollHeight;
        typingTimeout = setTimeout(type, speed);
      } else {
        isTyping = false;
        sendBtn.textContent = "➤";
        addFeedbackButtons(msgDiv, text);
        resolve();
      }
    }
    type();
  });
}

function showTyping() {
  const typing = document.createElement("div");
  typing.classList.add("message", "assistant-message");
  typing.id = "typing";
  typing.textContent = "Typing...";
  chatBox.appendChild(typing);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTyping() {
  const typing = document.getElementById("typing");
  if (typing) typing.remove();
}

function stopTyping() {
  if (isTyping) {
    clearTimeout(typingTimeout);
    if (controller) controller.abort();
    isTyping = false;
    sendBtn.textContent = "➤";
    removeTyping();
    appendMessage("⚠️ Response stopped by user.", "assistant");
  }
}

// ------------------------------
// Send Message
// ------------------------------
async function sendMessage() {
  const question = userInput.value.trim();
  if (!question) return;

  // Portal shortcut
  if (question.toLowerCase().includes("open portal")) {
    window.open("https://sp.srmist.edu.in/srmiststudentportal/students/loginManager/youLogin.jsp", "_blank");
    appendMessage("✅ Opening SRM Student Portal...", "assistant");
    userInput.value = "";
    return;
  }
  //Team Shortcut
  if (question.toLowerCase().includes("created you")) {
    window.open("/team", "_self"); 
    appendMessage("Meet the team", "assistant");
    userInput.value = "";
    return;
  }
  if (question.toLowerCase().includes("your code")) {
    appendMatrixMessage("🟩 Well, well… look who’s curious. Kudos, genius.", "assistant");
    startMatrixEffect();
    userInput.value = "";
    return;
}
  if (!isTyping) {
    appendMessage(question, "user");
    userInput.value = "";
    isTyping = true;
    sendBtn.textContent = "⏹";
    showTyping();

    controller = new AbortController();
    const signal = controller.signal;

    try {
    const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal
    });

    const data = await response.json(); // ONLY ONE CALL
    removeTyping();

    // Render timetable if exists
    if (data.timetable && data.timetable.length > 0) {
        renderTimetable(data.timetable);
    } else {
        await typeEffect(data.answer);
    }

} catch (err) {
  removeTyping();
  appendMessage(err.name === "AbortError" ? "⚠️ Response stopped by user." : "⚠️ Something went wrong.", "assistant");
  isTyping = false;
  sendBtn.textContent = "➤";
}

  } else {
    stopTyping();
  }
}

// ------------------------------
// Event Listeners
// ------------------------------
document.addEventListener("DOMContentLoaded", () => {
  const sendBtn = document.getElementById("send-btn");
  const userInput = document.getElementById("user-input");
  const portalBtn = document.getElementById("portalBtn");

  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keydown", e => { if (e.key === "Enter") sendMessage(); });

  portalBtn.addEventListener("click", () => {
    window.open("https://sp.srmist.edu.in/srmiststudentportal/students/loginManager/youLogin.jsp", "_blank");
  });
});


// Timetable Upload
document.getElementById("uploadTimetableBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("timetableFile");
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a file first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/upload_timetable", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  document.getElementById("timetableStatus").textContent = data.message;
});

// Seat Upload
document.getElementById("uploadSeatBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("seatFile");
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a file first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/upload_seats", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  document.getElementById("seatStatus").textContent = data.message;
});


// ------------------------------
// ------------------------------
// Admin Upload Password Protection
// ------------------------------
const adminPassword = "SRMAdmin2025"; // change to your secret password

document.getElementById("verifyPasswordBtn").addEventListener("click", () => {
  const inputPass = document.getElementById("adminPassword").value.trim();
  const status = document.getElementById("passwordStatus");
  const timetableDiv = document.getElementById("timetableUploadDiv");
  const seatDiv = document.getElementById("seatUploadDiv");
  const newsUploadDiv = document.getElementById("newsUploadDiv");

if (inputPass === adminPassword) {
  status.textContent = "✅ Password correct! You can now upload files.";
  status.style.color = "green";

  // Show upload divs
  [timetableDiv, seatDiv].forEach(div => {
    div.style.display = "block";
    div.style.opacity = 0;
    setTimeout(() => div.style.opacity = 1, 100);
    div.style.transition = "opacity 0.8s ease";
  });

  // Show Clear Data div
  const clearDiv = document.getElementById("clearDataDiv");
  clearDiv.style.display = "block";
  clearDiv.style.opacity = 0;
  setTimeout(() => clearDiv.style.opacity = 1, 100);
  clearDiv.style.transition = "opacity 0.8s ease";

    newsUploadDiv.style.display = "block";
    newsUploadDiv.style.opacity = 0;
    setTimeout(() => newsUploadDiv.style.opacity = 1, 100);
    newsUploadDiv.style.transition = "opacity 0.8s ease";
} else {
  status.textContent = "❌ Incorrect password. Access denied!";
  status.style.color = "red";
  timetableDiv.style.display = "none";
  seatDiv.style.display = "none";

  const clearDiv = document.getElementById("clearDataDiv");
  clearDiv.style.display = "none";
}
});

// Upload Timetable CSV
document.getElementById("uploadTimetableBtn").addEventListener("click", () => {
  const file = document.getElementById("timetableFile").files[0];
  const status = document.getElementById("timetableStatus");
  if (!file) return status.textContent = "Please select a file!";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("password", document.getElementById("adminPassword").value);

  fetch("/upload_timetable", {
    method: "POST",
    body: formData
  }).then(res => res.json())
    .then(data => status.textContent = data.message)
    .catch(err => status.textContent = "Upload failed!");
});

// Upload Seat CSV
document.getElementById("uploadSeatBtn").addEventListener("click", () => {
  const file = document.getElementById("seatFile").files[0];
  const status = document.getElementById("seatStatus");
  if (!file) return status.textContent = "Please select a file!";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("password", document.getElementById("adminPassword").value);

  fetch("/upload_seat", {
    method: "POST",
    body: formData
  }).then(res => res.json())
    .then(data => status.textContent = data.message)
    .catch(err => status.textContent = "Upload failed!");
});

// Admin Panel Dynamic Appearance
const adminPanel = document.getElementById("adminPanel");

window.addEventListener("scroll", function() {
  const scrollPosition = window.scrollY + window.innerHeight;
  const pageHeight = document.documentElement.scrollHeight;

  if (scrollPosition >= pageHeight - 100) {
    adminPanel.classList.add("visible");
  } else {
    adminPanel.classList.remove("visible");
  }
});

// Display chosen file name
document.getElementById("timetableFile").addEventListener("change", (e) => {
  document.getElementById("timetableFileName").textContent =
    e.target.files[0] ? e.target.files[0].name : "No file chosen";
});

document.getElementById("seatFile").addEventListener("change", (e) => {
  document.getElementById("seatFileName").textContent =
    e.target.files[0] ? e.target.files[0].name : "No file chosen";
});

window.addEventListener("scroll", function() {
  const adminPanel = document.querySelector(".adminPanel");
  const scrollPosition = window.scrollY + window.innerHeight;
  const pageHeight = document.documentElement.scrollHeight;

  // When near bottom → show it
  if (scrollPosition >= pageHeight - 100) {
    adminPanel.classList.add("visible");
  } 
  // When scrolled up → hide it
  else {
    adminPanel.classList.remove("visible");
  }
});
// ------------------------------
// Side Text Scroll Fade (Hide down, Show up)
// ------------------------------
const sideText = document.querySelector(".side-text");
let lastScrollTop = 0;

window.addEventListener("scroll", () => {
  const currentScroll = window.scrollY || document.documentElement.scrollTop;

  if (currentScroll > lastScrollTop) {
    // Scrolling down → hide
    sideText.style.opacity = "0";
    sideText.style.transform = "translateY(-60%)";
  } else {
    // Scrolling up → show
    sideText.style.opacity = "1";
    sideText.style.transform = "translateY(-50%)";
  }

  sideText.style.transition = "opacity 0.4s ease, transform 0.4s ease";
  lastScrollTop = currentScroll <= 0 ? 0 : currentScroll;
});


function clearData(type) {
    if (!confirm(`Are you sure you want to delete all ${type} data?`)) return;

    const formData = new FormData();
    formData.append("type", type);

    fetch("/clear_data", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("clearStatus").innerText = data.message;
    })
    .catch(err => {
        document.getElementById("clearStatus").innerText = "Error clearing data: " + err;
    });
}


  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("teamBtn").addEventListener("click", () => {
      window.location.href = "team.html";
    });
  });

document.addEventListener("DOMContentLoaded", () => {
    const tcModal = document.getElementById("tcModal");
    const acceptBtn = document.getElementById("acceptBtn");
    const teamBtn = document.getElementById("teamBtn");

    // Show modal / button based on localStorage
    if (localStorage.getItem("tcAccepted") === "true") {
        tcModal.style.display = "none";
        teamBtn.style.display = "block";
    } else {
        tcModal.style.display = "flex";  // flex for centering modal
        teamBtn.style.display = "none";
    }

    // Accept T&C
    acceptBtn.addEventListener("click", () => {
        localStorage.setItem("tcAccepted", "true");
        tcModal.style.display = "none";
        teamBtn.style.display = "block";
    });

    // Navigate to team page
    teamBtn.addEventListener("click", () => {
        window.location.href = "/team"; 
    });
});

function startMatrixEffect() {
    const canvas = document.getElementById("matrixCanvas");
    canvas.style.display = "block";
    const ctx = canvas.getContext("2d");

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()";
    const fontSize = 18;
    const columns = Math.floor(canvas.width / fontSize);
    const drops = Array(columns).fill(0);

    function draw() {
        ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = "#00ff00";
        ctx.font = fontSize + "px monospace";

        for (let i = 0; i < drops.length; i++) {
            const text = letters[Math.floor(Math.random() * letters.length)];
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);

            if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
        }

        requestAnimationFrame(draw);
    }

    draw();

    // Stop effect after 10 seconds
    setTimeout(() => {
        canvas.style.display = "none";
    }, 10000);
}

function appendMatrixMessage(msg) {
  const chatBox = document.getElementById("chat-box");
  const message = document.createElement("div");
  message.classList.add("message", "matrix");
  message.textContent = msg;
  chatBox.appendChild(message);

  // Trigger fade-in
  setTimeout(() => {
    message.classList.add("show", "pulse");
  }, 50);

  // Optional: Fade-out after 4 seconds
  setTimeout(() => {
    message.classList.remove("show");
    setTimeout(() => message.remove(), 800); // remove after fade-out
  }, 10000);

  chatBox.scrollTop = chatBox.scrollHeight;
}

document.getElementById("verifyPasswordBtn").addEventListener("click", () => {
  const panel = document.querySelector(".admin-panel");
  panel.classList.add("expanded");
});

