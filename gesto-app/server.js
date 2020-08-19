const express = require('express')
var http = require('http');
const https = require('https');
const path = require('path')
const { get } = require('request')
const bodyParser = require('body-parser');
const fs = require('fs');
const device = require('device')

const app = express()
var img_count = 0;


const cert = fs.readFileSync('certs/biojuan_com.crt');
const ca = fs.readFileSync('certs/biojuan_com.ca-bundle');
const key = fs.readFileSync('certs/biojuan_com.key');

const httpsOptions = {cert, ca, key};
const httpsServer = https.createServer(httpsOptions, app);


// Secondary http app
var httpApp = express();
var httpRouter = express.Router();
httpApp.use('*', httpRouter);
httpRouter.get('*', function(req, res){
    return res.redirect('https://biojuan.com');
});
var httpServer = http.createServer(httpApp);
httpServer.listen(80)

httpsServer.listen(443);


app.use(express.json({
  limit: '50mb',
  extended: true,
  parameterLimit:50000,
  type:'application/json'
}))

app.use(express.urlencoded({ extended: true }))

const viewsDir = path.join(__dirname, 'views')

app.use(express.static(viewsDir))
app.use(express.static(path.join(__dirname, './public')))
app.use(express.static(path.join(__dirname, './face-api.js/weights/')))
app.use(express.static(path.join(__dirname, './face-api.js/dist/')))

app.get('/', function (req, res)
	{
		res.redirect('/GESTOexperimento')
	}
)

app.get('/GESTOexperimento', function (req, res) { 
	var ua = device(req.headers['user-agent']);
	  if(ua.is('phone')){
		   res.sendFile(path.join(viewsDir, 'nodesktop.html'))
		}else if(ua.is('tablet')){
		   res.sendFile(path.join(viewsDir, 'nodesktop.html'))
		}else  if(ua.is('desktop')){
		   res.sendFile(path.join(viewsDir, 'experimento.html'))
		}else {
		    res.sendFile(path.join(viewsDir, 'nodesktop.html'))
		}
	}
)

app.get('/GESTOagradecimientos', function (req, res) { 
	
res.sendFile(path.join(viewsDir, 'agradecimientos.html'))}
)

app.post('/xml', function(request, response){
	var jsony = request.body
	var imgdata = jsony.img
	
	const base64Data = imgdata.replace(/^data:([A-Za-z-+/]+);base64,/, '');
	response.connection.destroy();	
	fs.writeFile('images/'+jsony.userid+jsony.gesture+jsony.number.toString()+'.webp', base64Data, 'base64', (err) => {

	});
	img_count++;
});

app.post('/GESTOexperimento', function(request, response){

	var userinfo = request.body
	var userid = (userinfo.firstname + userinfo.lastname).toLowerCase().replace(/\s/g, '').concat(pad(Math.floor(Math.random()*10000),4))
	
	var userinfo = {
						'firstname':request.body.firstname,
						'lastname':request.body.lastname,
						'email':request.body.email,
						'cellphone':request.body.cellphone,
						'userid':userid
					};
					
	fs.writeFile('jsons/'+userid+".json", JSON.stringify(userinfo), 'utf8', function (err) {
    
	if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
    }
    console.log("JSON file has been saved.");
	});
	 response.writeHead(200);
	 response.end(userid);
});


function request(url, returnBuffer = true, timeout = 10000) {
  return new Promise(function(resolve, reject) {
    const options = Object.assign(
      {},
      {
        url,
        isBuffer: true,
        timeout,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
        }
      },
      returnBuffer ? { encoding: null } : {}
    )

    get(options, function(err, res) {
      if (err) return reject(err)
      return resolve(res)
    })
  })
}

 function pad(n, width) { 
                n = n + ''; 
                return n.length >= width ? n :  
                    new Array(width - n.length + 1).join('0') + n; 
            } 