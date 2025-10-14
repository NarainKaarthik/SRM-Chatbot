// ------------------------------
// DOM Elements
// ------------------------------
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const portalBtn = document.getElementById("portalBtn");
const timetable = document.getElementById("exam-timetable");

let isTyping = false;
let typingTimeout;
let controller;

// ------------------------------
// Mouse Parallax Effect
// ------------------------------
document.addEventListener("mousemove", (e) => {
  const mouseEffect = document.querySelector(".mouse-effect");
  mouseEffect.style.background = `radial-gradient(circle at ${e.clientX}px ${e.clientY}px, rgba(251,175,65,0.05), transparent 25%)`;
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


// Check if timetable data exists
if (data.timetable && data.timetable.length > 0) {
    const tbody = document.querySelector("#timetable-table tbody");
    tbody.innerHTML = ""; // clear old rows
    data.timetable.forEach(row => {
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
    document.getElementById("exam-timetable").style.display = "block";
     // show table
} else {
    await typeEffect(data.answer); // normal chat message
}
}

// ------------------------------
// Event Listeners
// ------------------------------
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", e => { if (e.key === "Enter") sendMessage(); });
portalBtn.addEventListener("click", () => {
  window.open("https://sp.srmist.edu.in/srmiststudentportal/students/loginManager/youLogin.jsp", "_blank");
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


//Timetable Section Toggle
fetch("/ask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: userQuestion})
})
.then(res => res.json())
.then(data => {
    // If timetable data exists
    if (data.timetable && data.timetable.length > 0) {
    const tbody = document.querySelector("#timetable-table tbody");
    tbody.innerHTML = ""; // clear old rows
    data.timetable.forEach(row => {
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
    document.getElementById("timetable-section").style.display = "block"; // show table
} else {
    // Normal chat message
    document.getElementById("chat-box").innerHTML += `<p>${data.answer}</p>`;
}
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
const sideText = document.querySelector(".side-text");
const header = document.querySelector(".srm-header");
const fadeDistance = 200; // pixels before reaching header to start fade

window.addEventListener("scroll", () => {
  const scrollY = window.scrollY;
  const headerBottom = header.offsetTop + header.offsetHeight;

  // Start fade 200px before header
  const startFade = headerBottom + 50; 
  const endFade = headerBottom; 

  if (scrollY < headerBottom - fadeDistance) {
    sideText.style.opacity = 1;
    sideText.style.transform = `translateY(-50%)`;
  } else if (scrollY >= startFade) {
    const progress = Math.max(0, (scrollY - endFade) / fadeDistance);
    sideText.style.opacity = 1 - progress;
    sideText.style.transform = `translateY(${ -50 - (progress * 30) }%)`;
  } else {
    sideText.style.opacity = 0;
    sideText.style.transform = `translateY(-80%)`;
  }
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

