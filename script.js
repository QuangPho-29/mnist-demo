// Tạo canvas 280x280 nhưng chia thành lưới 28x28
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
// const processedCanvas = document.getElementById('preprocessed_canvas');
// const processedCtx = processedCanvas.getContext('2d');

// Kích thước
const gridSize = 28;
const cellSize = 10; // Kích thước mỗi ô lưới (10x10 pixels)
canvas.width = gridSize * cellSize;
canvas.height = gridSize * cellSize;
// processedCanvas.width = gridSize;
// processedCanvas.height = gridSize;

// Nút và kết quả
const predictButton = document.getElementById('predict');
const clearButton = document.getElementById('clear');
const result = document.getElementById('result');

// Đặt nền đen cho canvas
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Trạng thái vẽ
let isDrawing = false;

// Sự kiện vẽ
canvas.addEventListener('mousedown', () => (isDrawing = true));
canvas.addEventListener('mouseup', () => (isDrawing = false));
canvas.addEventListener('mousemove', draw);

function draw(event) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / cellSize);
    const y = Math.floor((event.clientY - rect.top) / cellSize);

    // Áp dụng brush 3x3
    const brushOffsets = [
        { dx: -1, dy: -1, opacity: 0.5 }, // Ô 1
        { dx: 0, dy: -1, opacity: 0.8 },  // Ô 2
        { dx: 1, dy: -1, opacity: 0.5 },  // Ô 3
        { dx: -1, dy: 0, opacity: 0.8 },  // Ô 4
        { dx: 0, dy: 0, opacity: 1.0 },   // Ô 5 (giữa)
        { dx: 1, dy: 0, opacity: 0.8 },   // Ô 6
        { dx: -1, dy: 1, opacity: 0.5 },  // Ô 7
        { dx: 0, dy: 1, opacity: 0.8 },   // Ô 8
        { dx: 1, dy: 1, opacity: 0.5 },   // Ô 9
    ];

    ctx.globalCompositeOperation = 'source-over';
    brushOffsets.forEach(({ dx, dy, opacity }) => {
        const nx = x + dx;
        const ny = y + dy;

        // Chỉ vẽ nếu ô nằm trong lưới
        if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize) {
            ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
            ctx.fillRect(nx * cellSize, ny * cellSize, cellSize, cellSize);
        }
    });
}

// Xóa canvas
clearButton.addEventListener('click', () => {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // processedCtx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
    resetBars();
    result.textContent = '';

});

// Reset các thanh bar
function resetBars() {
    for (let i = 0; i < 10; i++) {
        const bar = document.getElementById(`bar${i}`);
        const number = document.getElementById(`number${i}`);

        // Đặt chiều cao của thanh về 0%
        bar.style.height = '0%';

        // Xóa thuộc tính data-content
        bar.setAttribute('bar-content', '');

        // Loại bỏ trạng thái active
        bar.classList.remove('active');
        number.classList.remove('active');
    }
}

// Xử lý ảnh và hiển thị lên canvas 28x28
function preprocessCanvas() {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const grayImage = new Uint8ClampedArray(gridSize * gridSize * 4);

    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            const idx = (y * gridSize + x) * 4;
            const value = ctx.getImageData(x * cellSize, y * cellSize, cellSize, cellSize).data[0];
            grayImage[idx] = value;
            grayImage[idx + 1] = value;
            grayImage[idx + 2] = value;
            grayImage[idx + 3] = 255;
        }
    }

    const imageData28x28 = new ImageData(grayImage, gridSize, gridSize);
    // processedCtx.putImageData(imageData28x28, 0, 0);

    return Array.from(grayImage).filter((_, i) => i % 4 === 0).map(v => v / 255.0);
}

// Load ONNX model
let session;
async function loadModel() {
    session = await ort.InferenceSession.create('./mnist_model.onnx');
    console.log('Mô hình đã được tải');
}

function softmax(logits) {
    // Tính e^x cho mỗi phần tử và chuẩn hóa
    const expValues = logits.map(x => Math.exp(x)); // Lấy e^logit
    const sumExp = expValues.reduce((a, b) => a + b, 0); // Tổng tất cả e^logit
    return expValues.map(x => x / sumExp); // Chuẩn hóa thành xác suất
}

// Dự đoán số
async function predict() {
    const pixelData = preprocessCanvas();
    const tensor = new ort.Tensor('float32', new Float32Array(pixelData), [1, 1, 28, 28]);

    const feeds = { input: tensor };
    const output = await session.run(feeds);
    // console.log(output); // Kiểm tra toàn bộ cấu trúc đầu ra


    // Kiểm tra và trích xuất logits
    const logits = Object.values(output.output.cpuData);
    if (!Array.isArray(logits)) {
        console.error("Logits không hợp lệ:", logits);
        return;
    }

    const probabilities = softmax(logits); // Chuyển logits thành xác suất

    // console.log(probabilities); // Hiển thị xác suất

    const digit = probabilities.indexOf(Math.max(...probabilities)); // Lớp có xác suất cao nhất
    result.textContent = `${digit}`;

    updateBars(probabilities, digit); // Cập nhật các thanh bars với xác suất
}


function updateBars(probabilities, digit) {
    probabilities.forEach((value, index) => {
        const bar = document.getElementById(`bar${index}`);
        const number = document.getElementById(`number${index}`);

        if (value > 0.01) {
            // Cập nhật chiều cao của thanh
            bar.style.height = `${value * 100}%`; // Hiển thị phần trăm xác suất
            // console.log(bar);

            // Cập nhật nội dung chỉ khi giá trị > 0
            bar.setAttribute('bar-content', `${(value * 100).toFixed(2)}%`);
        } else {
            // Nếu giá trị ≤ 1, ẩn nội dung và đặt chiều cao về 0
            bar.style.height = '0%';
            bar.setAttribute('bar-content', '');
        }

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


// Tải mô hình khi trang load
loadModel();
predictButton.addEventListener('click', predict);
