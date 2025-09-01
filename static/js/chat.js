document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    const chatForm = document.getElementById('chat-form');
    const questionInput = document.getElementById('question');
    const messageWindow = document.getElementById('message-window');
    const sendButton = document.getElementById('send-button');
    const micButton = document.getElementById('mic-button');

    let mediaRecorder;
    let audioChunks = [];
    let recordingTimeout;

    micButton.addEventListener('mousedown', async () => {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    clearTimeout(recordingTimeout); // Clear timeout if stopped manually
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
                    audioChunks = [];
                    const formData = new FormData();
                    formData.append('file', audioBlob, 'recording.webm');

                    try {
                        const response = await fetch('/api/transcribe', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();
                        if (data.transcript) {
                            questionInput.value = data.transcript;
                        }
                    } catch (error) {
                        console.error('Error transcribing audio:', error);
                    }
                };

                mediaRecorder.start();
                micButton.classList.add('recording');

                // Set 15-second timeout
                recordingTimeout = setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                        micButton.classList.remove('recording');
                    }
                }, 15000); // 15 seconds

            } catch (error) {
                console.error('Error accessing microphone:', error);
            }
        }
    });

    micButton.addEventListener('mouseup', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            micButton.classList.remove('recording');
        }
    });

    // Apply cached theme on load
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        body.classList.add(savedTheme);
        if (savedTheme === 'light-mode') {
            themeToggle.checked = true;
        }
    } else {
        // Default to dark mode if no theme is saved
        body.classList.remove('light-mode');
        localStorage.setItem('theme', '');
    }

    // Theme switcher event listener
    themeToggle.addEventListener('change', () => {
        if (themeToggle.checked) {
            body.classList.add('light-mode');
            localStorage.setItem('theme', 'light-mode');
        } else {
            body.classList.remove('light-mode');
            localStorage.setItem('theme', '');
        }
    });

    // Auto-resize textarea
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = `${questionInput.scrollHeight}px`;
    });

    // Send on Enter, new line on Ctrl+Enter
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.ctrlKey) {
            e.preventDefault();
            chatForm.requestSubmit();
        }
    });

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const question = questionInput.value.trim();
        if (!question) return;

        // Disable form
        questionInput.value = '';
        questionInput.style.height = 'auto';
        questionInput.disabled = true;
        sendButton.disabled = true;

        // Display user message
        appendMessage(question, 'user');

        // Display bot message container
        const botMessageDiv = appendMessage('', 'bot');
        const bubbleDiv = botMessageDiv.querySelector('.message-bubble');
        bubbleDiv.innerHTML = '<div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div>';
        botMessageDiv.classList.add('loading');
        
        const eventSource = new EventSource(`/api/ask?question=${encodeURIComponent(question)}`);
        
        let fullAnswer = '';

        eventSource.addEventListener('answer_token', (e) => {
            // Remove loading indicator on first token
            if (botMessageDiv.classList.contains('loading')) {
                botMessageDiv.classList.remove('loading');
                bubbleDiv.innerHTML = '';
            }
            
            const data = JSON.parse(e.data);
            fullAnswer += data.token;
            bubbleDiv.textContent = fullAnswer;
            scrollToBottom();
        });

        eventSource.addEventListener('stream_end', (e) => {
            eventSource.close();
            // Re-enable form
            questionInput.disabled = false;
            sendButton.disabled = false;
            questionInput.focus();
            
            // Add feedback buttons
            const feedbackContainer = document.createElement('div');
            feedbackContainer.classList.add('feedback-container');
            feedbackContainer.innerHTML = `
                <button class="feedback-btn like" data-rating="1"><i class="fas fa-thumbs-up"></i></button>
                <button class="feedback-btn dislike" data-rating="0"><i class="fas fa-thumbs-down"></i></button>
                <button class="feedback-btn speak" title="Ouvir resposta"><i class="fas fa-volume-up"></i></button>
            `;
            bubbleDiv.appendChild(feedbackContainer);

            feedbackContainer.querySelector('.speak').addEventListener('click', (event) => {
                synthesizeAndPlay(fullAnswer, event.currentTarget);
            });

            feedbackContainer.querySelectorAll('.feedback-btn:not(.speak)').forEach(btn => {
                btn.addEventListener('click', (event) => {
                    const rating = event.currentTarget.dataset.rating;
                    sendFeedback(question, fullAnswer, rating);
                    feedbackContainer.querySelectorAll('.feedback-btn:not(.speak)').forEach(button => button.disabled = true);
                    event.currentTarget.style.transform = 'scale(1.2)';
                });
            });

            scrollToBottom();
        });

        eventSource.onerror = (err) => {
            console.error('EventSource failed:', err);
            eventSource.close();
            bubbleDiv.textContent = 'Desculpe, ocorreu um erro ao processar sua pergunta.';
            botMessageDiv.classList.remove('loading');
            // Re-enable form
            questionInput.disabled = false;
            sendButton.disabled = false;
            questionInput.focus();
        };
    });

    function appendMessage(text, type, isLoading = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);

        const bubbleDiv = document.createElement('div');
        bubbleDiv.classList.add('message-bubble');

        if (isLoading) {
            bubbleDiv.innerHTML = '<div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div>';
            messageDiv.classList.add('loading');
        } else {
            bubbleDiv.textContent = text;
        }

        messageDiv.appendChild(bubbleDiv);
        messageWindow.appendChild(messageDiv);
        scrollToBottom();
        return messageDiv;
    }

    function updateMessage(messageDiv, newText, messageId) {
        messageDiv.classList.remove('loading');
        const bubbleDiv = messageDiv.querySelector('.message-bubble');
        bubbleDiv.innerHTML = ''; // Clear loading dots
        bubbleDiv.textContent = newText;

        if (messageDiv.classList.contains('bot')) {
            const feedbackContainer = document.createElement('div');
            feedbackContainer.classList.add('feedback-container');
            feedbackContainer.innerHTML = `
                <button class="feedback-btn like" data-id="${messageId}" data-rating="1"><i class="fas fa-thumbs-up"></i></button>
                <button class="feedback-btn dislike" data-id="${messageId}" data-rating="0"><i class="fas fa-thumbs-down"></i></button>
            `;
            bubbleDiv.appendChild(feedbackContainer);

            feedbackContainer.querySelectorAll('.feedback-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const rating = e.currentTarget.dataset.rating;
                    const question = messageDiv.previousElementSibling.querySelector('.message-bubble').textContent;
                    const answer = newText;
                    
                    sendFeedback(question, answer, rating);
                    
                    // Disable buttons after click
                    feedbackContainer.querySelectorAll('.feedback-btn').forEach(button => button.disabled = true);
                    e.currentTarget.style.transform = 'scale(1.2)';
                });
            });
        }
        scrollToBottom();
    }

    async function sendFeedback(question, answer, rating) {
        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question, answer, rating }),
            });
        } catch (error) {
            console.error('Error sending feedback:', error);
        }
    }

    async function synthesizeAndPlay(text, button) {
        const icon = button.querySelector('i');
        button.disabled = true;
        icon.classList.remove('fa-volume-up', 'fa-exclamation-circle');
        icon.classList.add('fa-spinner', 'fa-spin');

        try {
            const response = await fetch('/api/synthesize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            
            audio.addEventListener('ended', () => {
                button.disabled = false;
                icon.classList.remove('fa-spinner', 'fa-spin');
                icon.classList.add('fa-volume-up');
                URL.revokeObjectURL(audioUrl); // Clean up
            });

            audio.play();

        } catch (error) {
            console.error('Error synthesizing speech:', error);
            button.disabled = false;
            icon.classList.remove('fa-spinner', 'fa-spin');
            icon.classList.add('fa-exclamation-circle');
        }
    }

    function scrollToBottom() {
        messageWindow.scrollTop = messageWindow.scrollHeight;
    }
});