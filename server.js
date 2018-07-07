const Brain = require('./tfLoader').Brain;


class Server {

  async init(){
    this.brain = new Brain();
    await this.brain.loadTensor("mobilenet");
  }

  async predict(b64data){
    const result = await this.brain.predictFromBase64(b64data);
    return result;
  }
}

const server = new Server();

module.exports = {server}