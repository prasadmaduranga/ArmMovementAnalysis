const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

videoElement.classList.toggle("selfie", true);
canvasElement.classList.toggle("selfie", true);

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);


  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                     {color: '#00FF00', lineWidth: 5});
      drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
    }
  }

  //   // begin custom shape
  // canvasCtx.beginPath();
  // canvasCtx.moveTo(170, 80);
  // canvasCtx.bezierCurveTo(130, 100, 130, 150, 230, 150);
  // canvasCtx.bezierCurveTo(250, 180, 320, 180, 340, 150);
  // canvasCtx.bezierCurveTo(420, 150, 420, 120, 390, 100);
  // canvasCtx.bezierCurveTo(430, 40, 370, 30, 340, 50);
  // canvasCtx.bezierCurveTo(320, 5, 250, 20, 250, 50);
  // canvasCtx.bezierCurveTo(200, 5, 150, 20, 170, 80);
  //
  // // complete custom shape
  // canvasCtx.closePath();
  // canvasCtx.lineWidth = 5;
  // canvasCtx.fillStyle = 'rgba(102, 204, 255, 0.5)';
  // canvasCtx.fill();
  // canvasCtx.strokeStyle = 'blue';
  // canvasCtx.stroke();
//========
  // canvasCtx.restore();
  //
  //
      function drawCircle(x, y, radiusX, radiusY, rotation) {
     canvasCtx.lineWidth = 5;
        canvasCtx.fillStyle = 'rgba(102, 204, 255, 0)';
        canvasCtx.strokeStyle = 'blue';

        canvasCtx.beginPath();
        canvasCtx.ellipse(x, y, radiusX, radiusY, rotation, 0, 2 * Math.PI);
        canvasCtx.stroke();
        canvasCtx.fill();
    }

    var deg = 0;
    var rad = deg * (Math.PI / 180.0);

    // Eclipse paramters:
    // x=100; y=100;
    // width=80; height=40;
    // clockwiseRotation=20 degrees // rotation according to point (x, y)
    drawCircle(500, 150, 200, 100, rad);


}

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});


hands.onResults(onResults);


const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();