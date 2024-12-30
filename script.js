// Tạo canvas 28x28 nhưng hiển thị với mỗi pixel phóng to
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scaleFactor = 10;
canvas.width = 28 * scaleFactor;
canvas.height = 28 * scaleFactor;

// Nút và kết quả
const predictButton = document.getElementById('predict');
const clearButton = document.getElementById('clear');
const result = document.getElementById('result');

// Đặt nền đen ban đầu cho canvas
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Biến trạng thái vẽ
let isDrawing = false;

// Sự kiện vẽ
canvas.addEventListener('mousedown', () => (isDrawing = true));
canvas.addEventListener('mouseup', () => (isDrawing = false));
canvas.addEventListener('mousemove', draw);

function draw(event) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / scaleFactor) * scaleFactor;
    const y = Math.floor((event.clientY - rect.top) / scaleFactor) * scaleFactor;

    ctx.fillStyle = 'white';
    ctx.fillRect(x, y, scaleFactor, scaleFactor); // Vẽ từng pixel phóng to
}

// Clear canvas
clearButton.addEventListener('click', () => {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetBars();
});

// Reset các thanh bar
function resetBars() {
    for (let i = 0; i < 10; i++) {
        const bar = document.getElementById(`bar${i}`);
        bar.style.height = '0%';
        bar.classList.remove('active');

        const number = document.getElementById(`number${i}`);
        number.classList.remove('active');
    }
}


// Load mô hình ONNX
let session;
async function loadModel() {
    session = await ort.InferenceSession.create('./mnist_model.onnx');
    console.log('Mô hình đã được tải');
}

// Dự đoán
async function predict() {
    // Chuyển đổi ảnh từ canvas 280x280 về kích thước gốc 28x28
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const tensor = preprocessImage(imageData);

    // Chạy dự đoán
    const feeds = { input: tensor };
    const output = await session.run(feeds);

    // Lấy kết quả
    const predictions = output.output.data;
    const digit = predictions.indexOf(Math.max(...predictions));

    result.textContent = digit;

    // Cập nhật các thanh bar và số dự đoán
    updateBars(predictions, digit);
}

// Cập nhật thanh bar và màu số
function updateBars(predictions, digit) {
    predictions.forEach((probability, index) => {
        const bar = document.getElementById(`bar${index}`);
        const number = document.getElementById(`number${index}`);

        // Tính toán chiều rộng thanh bar dựa trên xác suất
        const percentage = (probability * 100).toFixed(2);
        bar.style.height = `${percentage}%`;

        // Cập nhật trạng thái active
        if (index === digit) {
            bar.classList.add('active');
            number.classList.add('active');
        } else {
            bar.classList.remove('active');
            number.classList.remove('active');
        }
    });
}


// Xử lý ảnh: giảm kích thước và chuẩn hóa
function preprocessImage(imageData) {
    const { data, width, height } = imageData;
    const grayImage = new Float32Array(28 * 28);

    for (let i = 0; i < 28; i++) {
        for (let j = 0; j < 28; j++) {
            let total = 0;
            let count = 0;

            // Lấy giá trị trung bình từ mỗi ô 10x10 pixel
            for (let x = 0; x < scaleFactor; x++) {
                for (let y = 0; y < scaleFactor; y++) {
                    const idx = ((i * scaleFactor + x) * width + (j * scaleFactor + y)) * 4;
                    total += data[idx]; // Lấy giá trị kênh đỏ (R) vì ảnh là grayscale
                    count++;
                }
            }

            grayImage[i * 28 + j] = (total / count) / 255.0; // Chuẩn hóa về [0, 1]
        }
    }

    // Trả về tensor [1, 1, 28, 28]
    return new ort.Tensor('float32', grayImage, [1, 1, 28, 28]);
}

// Tải mô hình khi trang load
loadModel();

// Thêm sự kiện cho nút "Dự đoán"
predictButton.addEventListener('click', predict);
