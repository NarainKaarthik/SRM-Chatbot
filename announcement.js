window.addEventListener("DOMContentLoaded", async () => {
  const container = document.getElementById("alertsSidebar");
  if (!container) return console.error("alertsSidebar not found!");

  let announcements = [];
  let currentIndex = 0;
  const batchSize = 3;

  async function fetchAnnouncements() {
    try {
      const res = await fetch("/get_announcements");
      announcements = await res.json();
      console.log("Fetched announcements:", announcements);
      if (!announcements.length) {
        container.innerHTML = "<p>No announcements yet.</p>";
        return;
      }
      loadNextBatch();
    } catch (err) {
      console.error("Error fetching:", err);
      container.innerHTML = "<p>Could not load announcements.</p>";
    }
  }

  function loadNextBatch() {
    const nextBatch = announcements.slice(currentIndex, currentIndex + batchSize);
    nextBatch.forEach(news => {
      const div = document.createElement("div");
      div.className = "announcement";
      div.innerHTML = `
        <h4>${news.title}</h4>
        <p>${news.message}</p>
        <small>${news.date}${news.urgent ? " 🔴 Urgent" : ""}</small>
      `;
      container.appendChild(div);
    });
    currentIndex += batchSize;

    // If reached end, start from beginning (infinite loop)
    if (currentIndex >= announcements.length) {
      currentIndex = 0;
    }
  }

  // Auto-scroll + infinite loop
  function startInfiniteScroll() {
    setInterval(() => {
      container.scrollBy({ top: 1, behavior: "smooth" });
      if (container.scrollTop + container.clientHeight >= container.scrollHeight) {
        container.scrollTop = 0; // loop
      }
    }, 50);
  }

  await fetchAnnouncements();
  startInfiniteScroll();
});

// ------- ADMIN: ADD NEWS -------
document.getElementById("addNewsBtn")?.addEventListener("click", async () => {
  const title = document.getElementById("newsTitle").value.trim();
  const message = document.getElementById("newsMessage").value.trim();
  const date = document.getElementById("newsDate").value;
  const urgent = document.getElementById("newsUrgent").checked;

  if (!title || !message || !date) {
    alert("Please fill all fields!");
    return;
  }

  try {
    const response = await fetch("/admin/add_news", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, message, date, urgent })
    });

    const data = await response.json();

    if (data.status === "success") {
      alert("News added successfully!");
      document.getElementById("newsTitle").value = "";
      document.getElementById("newsMessage").value = "";
      document.getElementById("newsDate").value = "";
      document.getElementById("newsUrgent").checked = false;

      // ✅ Refresh the right panel instantly
      if (window.loadAnnouncements) {
        window.loadAnnouncements();
      }
    } else {
      alert(data.message || "Failed to add news!");
    }
  } catch (err) {
    console.error("Error adding news:", err);
    alert("Error connecting to backend!");
  }
});

// ------- LOAD NEWS TITLES INTO DELETE DROPDOWN -------
async function loadNewsForDelete() {
  try {
    const response = await fetch("/get_announcements");
    const data = await response.json();

    const select = document.getElementById("delete-news-select");
    select.innerHTML = '<option value="">-- Select News to Delete --</option>';

    data.forEach(news => {
      const option = document.createElement("option");
      option.value = news.id;
      option.textContent = news.title;
      select.appendChild(option);
    });
  } catch (err) {
    console.error("Error loading news:", err);
  }
}

// ------- DELETE SELECTED NEWS -------
document.getElementById("delete-news-btn").addEventListener("click", async () => {
  const newsId = document.getElementById("delete-news-select").value;
  if (!newsId) {
    alert("Please select a news item to delete!");
    return;
  }

  if (!confirm("Are you sure you want to delete this news?")) return;

  try {
    const response = await fetch(`/admin/delete_news/${newsId}`, { method: "DELETE" });
    const result = await response.json();

    if (result.status === "success") {
      alert("News deleted successfully!");
      loadNewsForDelete(); // refresh dropdown
      if (window.loadAnnouncements) window.loadAnnouncements(); // refresh right panel
    } else {
      alert("Failed to delete news!");
    }
  } catch (err) {
    console.error("Error deleting news:", err);
    alert("Error connecting to backend!");
  }
});

// Call this when admin page loads
window.addEventListener("DOMContentLoaded", loadNewsForDelete);

