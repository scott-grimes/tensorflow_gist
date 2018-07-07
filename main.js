const tf = require('@tensorflow/tfjs');
const Brain = require('./tfLoader').Brain;


let predict;

// converts an image located at url into it's base64 encoded representation. can be a local file or other file as long as the origin allows CORS
var convertToBase64 = function (url) {
  return new Promise(function(resolve, reject) {
    //build a blank canvas, draw the image to it. then convert to base64 encoding for transfer to the server
    var img = document.createElement('IMG'),
      canvas = document.createElement('CANVAS'),
      ctx = canvas.getContext('2d'),
      data = '';
    //if we maybe want to use this with other image urls

    img.crossOrigin = 'Anonymous';


    img.onload = async function() {
      // when the image is loaded, *this is the image, encode it and resolve our solution
       canvas.height = this.height;
      canvas.width = this.width;
      ctx.drawImage(this, 0, 0);
      var imagetype = await getTypeFromImageUrl(url);
      data = canvas.toDataURL(imagetype);
      resolve(data);
    };

    // if it takes longer than 5sec to load, timeout
    setTimeout(()=>reject('timeout'), 5000);
    
    // start to load the image
    img.src = url;


  });

  
};


var sendBase64ToServer = function (base64) {

  addbase64ToPage(base64);
  return;

  var httpPost = new XMLHttpRequest(),
    path = 'http://127.0.0.1:8000/uploadImage/',
    data = JSON.stringify({ image: base64 });

  httpPost.onreadystatechange = function (err) {
    if (httpPost.readyState == 4 && httpPost.status == 200) {
      console.log(httpPost.responseText);
    } else {
      console.log(err);
    }
  };
  // Set the content type of the request to json since that's what's being sent
  httpPost.setHeader('Content-Type', 'application/json');
  httpPost.open('POST', path, true);
  httpPost.send(data);
};


var addbase64ToPage = function(data) {
  var image = new Image();
  image.src = data;
  document.body.appendChild(image);
  return image
};

const getTypeFromImageUrl =  async function getTypeFromImageUrl(imageUrl) {
  const response = await fetch(imageUrl)
  return response.blob().type;
}

// when an image is clicked, predict waht it is!
const predictFromClick = async (event)=>{
  if(!event) return;
  const image = event.target.src;
  
  const hd64 = await convertToBase64(image);
  const time = Date.now();
  console.log('Predicting...')
  const predictions = await predict(hd64);
  const duration = Date.now()-time;
  console.log(predictions)
  console.log(`Prediction completed in ${Math.floor(duration)}ms`);
  let hotdogToken = 'Not Hotdog'
  hotdogToken = predictions[0].className.includes('hotdog')? 'Maybe Hotdog' : hotdogToken;
  hotdogToken = predictions[0].probability > 0.9? 'Is Hotdog' : hotdogToken;
  let solution = 'Verdict: ';
  solution+= hotdogToken;
  solution+='\n\nMachine Thinks this is...\n';
  for(let predict of predictions){
    solution+=predict.className+' : '+Math.floor(Math.round( predict.probability*100))+'%\n'
  }
  console.log(solution)
  return solution
}


//console.log(mobilenet);
window.onload = async ()=>{

  const brain = new Brain();
  await brain.loadTensor("mobilenet");
  predict =  brain.predictFromBase64.bind(brain);
  var huskyimg = document.createElement('img')
  huskyimg.src ='husky.jpg'
  document.body.appendChild(huskyimg)
  var arr = document.getElementsByTagName('img');
  Array.prototype.slice.call(arr).forEach(i=>{
 
    i.addEventListener("click", predictFromClick);
  })

  
};


