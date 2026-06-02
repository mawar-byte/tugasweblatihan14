// ==========================================
// SECTION 1: HELPER FUNCTIONS & MATH ENGINE
// ==========================================

/**
 * Box-Muller transform to generate standard normal Gaussian values.
 */
function randomGaussian() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Segment-to-segment intersection checker.
 */
function intersectSegments(p1, p2, p3, p4) {
  const denominator = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
  if (denominator === 0) return null;

  const ua = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)) / denominator;
  const ub = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) / denominator;

  if (ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1) {
    return {
      x: p1.x + ua * (p2.x - p1.x),
      y: p1.y + ua * (p2.y - p1.y),
      t: ua
    };
  }
  return null;
}

/**
 * Ray to line intersection helper.
 */
function intersectRayToLine(p1, p2, p3, p4) {
  const denominator = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
  if (denominator === 0) return null;

  const ua = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)) / denominator;
  const ub = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) / denominator;

  if (ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1) {
    return {
      x: p1.x + ua * (p2.x - p1.x),
      y: p1.y + ua * (p2.y - p1.y),
      t: ua
    };
  }
  return null;
}

// ==========================================
// SECTION 2: NEURAL NETWORK LIBRARY
// ==========================================

class Layer {
  constructor(inputSize, outputSize, randomize = true) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights = [];
    this.biases = [];
    
    if (randomize) {
      this.randomize();
    } else {
      this.zeroInit();
    }
  }

  randomize() {
    const scale = Math.sqrt(2.0 / this.inputSize);
    for (let i = 0; i < this.outputSize; i++) {
      this.biases.push(randomGaussian() * 0.1);
      const row = [];
      for (let j = 0; j < this.inputSize; j++) {
        row.push(randomGaussian() * scale);
      }
      this.weights.push(row);
    }
  }

  zeroInit() {
    for (let i = 0; i < this.outputSize; i++) {
      this.biases.push(0);
      const row = new Array(this.inputSize).fill(0);
      this.weights.push(row);
    }
  }

  feedForward(inputs) {
    const outputs = [];
    for (let i = 0; i < this.outputSize; i++) {
      let sum = this.biases[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += inputs[j] * this.weights[i][j];
      }
      outputs.push(Math.tanh(sum));
    }
    return outputs;
  }

  mutate(rate, amount) {
    for (let i = 0; i < this.outputSize; i++) {
      if (Math.random() < rate) {
        this.biases[i] += randomGaussian() * amount;
      }
      for (let j = 0; j < this.inputSize; j++) {
        if (Math.random() < rate) {
          this.weights[i][j] += randomGaussian() * amount;
        }
      }
    }
  }

  copy() {
    const clone = new Layer(this.inputSize, this.outputSize, false);
    clone.biases = [...this.biases];
    clone.weights = this.weights.map(row => [...row]);
    return clone;
  }

  crossover(partner) {
    const child = new Layer(this.inputSize, this.outputSize, false);
    for (let i = 0; i < this.outputSize; i++) {
      child.biases[i] = Math.random() < 0.5 ? this.biases[i] : partner.biases[i];
      for (let j = 0; j < this.inputSize; j++) {
        child.weights[i][j] = Math.random() < 0.5 ? this.weights[i][j] : partner.weights[i][j];
      }
    }
    return child;
  }
}

class NeuralNetwork {
  constructor(layerSizes) {
    this.layerSizes = layerSizes;
    this.layers = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
      this.layers.push(new Layer(layerSizes[i], layerSizes[i + 1]));
    }
  }

  feedForward(inputs) {
    const activations = [inputs];
    let currentInputs = inputs;
    for (let i = 0; i < this.layers.length; i++) {
      currentInputs = this.layers[i].feedForward(currentInputs);
      activations.push(currentInputs);
    }
    return activations;
  }

  getOutput(inputs) {
    const activations = this.feedForward(inputs);
    return activations[activations.length - 1];
  }

  mutate(rate, amount) {
    for (let layer of this.layers) {
      layer.mutate(rate, amount);
    }
  }

  copy() {
    const clone = new NeuralNetwork(this.layerSizes);
    clone.layers = this.layers.map(layer => layer.copy());
    return clone;
  }

  crossover(partner) {
    const child = new NeuralNetwork(this.layerSizes);
    child.layers = this.layers.map((layer, index) => {
      return layer.crossover(partner.layers[index]);
    });
    return child;
  }
}

// ==========================================
// SECTION 3: OBSTACLES ENGINE
// ==========================================

class Obstacle {
  constructor(type) {
    this.type = type;
    this.vx = 0;
    this.vy = 0;
    this.minX = null;
    this.maxX = null;
    this.minY = null;
    this.maxY = null;
  }
  
  setMotion(vx, vy, minX, maxX, minY, maxY) {
    this.vx = vx;
    this.vy = vy;
    this.minX = minX;
    this.maxX = maxX;
    this.minY = minY;
    this.maxY = maxY;
  }

  update() {
    if (this.vx === 0 && this.vy === 0) return;
    
    this.x += this.vx;
    this.y += this.vy;
    
    if (this.minX !== null && this.x < this.minX) {
      this.x = this.minX;
      this.vx *= -1;
    } else if (this.maxX !== null && this.x > this.maxX) {
      this.x = this.maxX;
      this.vx *= -1;
    }
    
    if (this.minY !== null && this.y < this.minY) {
      this.y = this.minY;
      this.vy *= -1;
    } else if (this.maxY !== null && this.y > this.maxY) {
      this.y = this.maxY;
      this.vy *= -1;
    }
  }

  draw(ctx) {}
  collidesWithCircle(cx, cy, cr) { return false; }
  intersectRay(p1, p2) { return null; }
  containsPoint(px, py) { return false; }
}

class RectObstacle extends Obstacle {
  constructor(x, y, width, height) {
    super('rect');
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  }

  draw(ctx) {
    ctx.save();
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.5)';
    ctx.fillStyle = 'rgba(30, 9, 15, 0.7)';
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    ctx.rect(this.x, this.y, this.width, this.height);
    ctx.fill();
    ctx.stroke();
    
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.15)';
    ctx.lineWidth = 1;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    for (let offset = 5; offset < this.width + this.height; offset += 15) {
      const startX = Math.max(this.x, this.x + offset - this.height);
      const startY = Math.min(this.y + this.height, this.y + offset);
      const endX = Math.min(this.x + this.width, this.x + offset);
      const endY = Math.max(this.y, this.y + offset - this.width);
      if (startX < endX) {
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  collidesWithCircle(cx, cy, cr) {
    const closestX = Math.max(this.x, Math.min(cx, this.x + this.width));
    const closestY = Math.max(this.y, Math.min(cy, this.y + this.height));
    const dist = Math.hypot(cx - closestX, cy - closestY);
    return dist < cr;
  }

  containsPoint(px, py) {
    return px >= this.x && px <= this.x + this.width &&
           py >= this.y && py <= this.y + this.height;
  }

  intersectRay(p1, p2) {
    const r1 = { x: this.x, y: this.y };
    const r2 = { x: this.x + this.width, y: this.y };
    const r3 = { x: this.x + this.width, y: this.y + this.height };
    const r4 = { x: this.x, y: this.y + this.height };

    const edges = [[r1, r2], [r2, r3], [r3, r4], [r4, r1]];
    let closestIntersection = null;

    for (const edge of edges) {
      const intersect = intersectSegments(p1, p2, edge[0], edge[1]);
      if (intersect) {
        if (!closestIntersection || intersect.t < closestIntersection.t) {
          closestIntersection = intersect;
        }
      }
    }
    return closestIntersection;
  }
}

class CircleObstacle extends Obstacle {
  constructor(x, y, radius) {
    super('circle');
    this.x = x;
    this.y = y;
    this.radius = radius;
  }

  draw(ctx) {
    ctx.save();
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.5)';
    ctx.fillStyle = 'rgba(30, 9, 15, 0.7)';
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.15)';
    ctx.lineWidth = 1;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius * 0.6, 0, Math.PI * 2);
    ctx.moveTo(this.x - this.radius, this.y);
    ctx.lineTo(this.x + this.radius, this.y);
    ctx.moveTo(this.x, this.y - this.radius);
    ctx.lineTo(this.x, this.y + this.radius);
    ctx.stroke();
    ctx.restore();
  }

  collidesWithCircle(cx, cy, cr) {
    const dist = Math.hypot(cx - this.x, cy - this.y);
    return dist < (this.radius + cr);
  }

  containsPoint(px, py) {
    return Math.hypot(px - this.x, py - this.y) <= this.radius;
  }

  intersectRay(p1, p2) {
    const dX = p2.x - p1.x;
    const dY = p2.y - p1.y;
    const fX = p1.x - this.x;
    const fY = p1.y - this.y;

    const a = dX * dX + dY * dY;
    const b = 2 * (fX * dX + fY * dY);
    const c = (fX * fX + fY * fY) - this.radius * this.radius;

    const discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return null;

    const t1 = (-b - Math.sqrt(discriminant)) / (2 * a);
    const t2 = (-b + Math.sqrt(discriminant)) / (2 * a);

    if (t1 >= 0 && t1 <= 1) {
      return { x: p1.x + t1 * dX, y: p1.y + t1 * dY, t: t1 };
    }
    if (t2 >= 0 && t2 <= 1) {
      return { x: p1.x + t2 * dX, y: p1.y + t2 * dY, t: t2 };
    }
    return null;
  }
}

// ==========================================
// SECTION 4: BEE AGENT DRONE
// ==========================================

class Bee {
  constructor(startX, startY, brainTopology, existingBrain = null) {
    this.startX = startX;
    this.startY = startY;
    this.x = startX;
    this.y = startY;
    this.vx = 0;
    this.vy = 0;
    this.angle = -Math.PI / 2;
    this.radius = 8;
    
    this.maxSpeed = 4;
    this.maxForce = 0.25;
    this.maxSteer = 0.12;
    
    this.brainTopology = brainTopology;
    this.brain = existingBrain ? existingBrain.copy() : new NeuralNetwork(brainTopology);
    this.lastActivations = null;
    
    this.isDead = false;
    this.hasSucceeded = false;
    this.stepsTaken = 0;
    this.closestDist = Infinity;
    this.startDist = 0;
    this.fitness = 0;
    
    this.sensorAngles = [
      -Math.PI / 2,
      -Math.PI / 4,
      -Math.PI / 12,
      0,
      Math.PI / 12,
      Math.PI / 4,
      Math.PI / 2
    ];
    this.sensorLength = 160;
    this.sensorReadings = new Array(this.sensorAngles.length).fill(1.0);
    this.sensorPoints = new Array(this.sensorAngles.length).fill(null);
    
    this.trail = [];
    this.maxTrailLength = 20;
    this.wingPhase = Math.random() * Math.PI * 2;
  }

  update(obstacles, target, boundaryWidth, boundaryHeight, currentStep) {
    if (this.isDead || this.hasSucceeded) return;
    
    this.stepsTaken = currentStep;
    
    const distToTarget = Math.hypot(target.x - this.x, target.y - this.y);
    if (this.startDist === 0) {
      this.startDist = distToTarget;
    }
    if (distToTarget < this.closestDist) {
      this.closestDist = distToTarget;
    }
    
    if (distToTarget < this.radius + target.radius) {
      this.hasSucceeded = true;
      this.calculateFitness();
      return;
    }
    
    this.updateSensors(obstacles, boundaryWidth, boundaryHeight);
    
    const inputs = [];
    for (let reading of this.sensorReadings) {
      inputs.push(reading);
    }
    inputs.push((target.x - this.x) / distToTarget);
    inputs.push((target.y - this.y) / distToTarget);
    inputs.push(this.vx / this.maxSpeed);
    inputs.push(this.vy / this.maxSpeed);
    
    this.lastActivations = this.brain.feedForward(inputs);
    const outputs = this.lastActivations[this.lastActivations.length - 1];
    
    const steerForce = outputs[0] * this.maxSteer;
    const thrust = outputs[1];
    
    this.angle += steerForce;
    
    let accelX = Math.cos(this.angle) * thrust * this.maxForce;
    let accelY = Math.sin(this.angle) * thrust * this.maxForce;
    
    this.vx += accelX;
    this.vy += accelY;
    
    this.vx *= 0.97;
    this.vy *= 0.97;
    
    const speed = Math.hypot(this.vx, this.vy);
    if (speed > this.maxSpeed) {
      this.vx = (this.vx / speed) * this.maxSpeed;
      this.vy = (this.vy / speed) * this.maxSpeed;
    }
    
    this.x += this.vx;
    this.y += this.vy;
    
    this.trail.push({ x: this.x, y: this.y });
    if (this.trail.length > this.maxTrailLength) {
      this.trail.shift();
    }
    
    this.wingPhase += 0.4 + Math.abs(thrust) * 0.4;
    
    this.checkCollisions(obstacles, boundaryWidth, boundaryHeight);
    
    if (this.stepsTaken > 900) {
      this.isDead = true;
      this.calculateFitness();
    }
  }

  updateSensors(obstacles, boundaryWidth, boundaryHeight) {
    for (let i = 0; i < this.sensorAngles.length; i++) {
      const globalAngle = this.angle + this.sensorAngles[i];
      const startPoint = { x: this.x, y: this.y };
      const endPoint = {
        x: this.x + Math.cos(globalAngle) * this.sensorLength,
        y: this.y + Math.sin(globalAngle) * this.sensorLength
      };
      
      let minT = 1.0;
      let closestPoint = endPoint;
      
      const bounds = [
        { p3: { x: 0, y: 0 }, p4: { x: boundaryWidth, y: 0 } },
        { p3: { x: 0, y: boundaryHeight }, p4: { x: boundaryWidth, y: boundaryHeight } },
        { p3: { x: 0, y: 0 }, p4: { x: 0, y: boundaryHeight } },
        { p3: { x: boundaryWidth, y: 0 }, p4: { x: boundaryWidth, y: boundaryHeight } }
      ];
      
      for (let bound of bounds) {
        const d = intersectRayToLine(startPoint, endPoint, bound.p3, bound.p4);
        if (d && d.t < minT) {
          minT = d.t;
          closestPoint = d;
        }
      }
      
      const minX = Math.min(startPoint.x, endPoint.x);
      const maxX = Math.max(startPoint.x, endPoint.x);
      const minY = Math.min(startPoint.y, endPoint.y);
      const maxY = Math.max(startPoint.y, endPoint.y);
      
      for (let obs of obstacles) {
        if (obs.type === 'circle') {
          if (obs.x + obs.radius < minX || obs.x - obs.radius > maxX ||
              obs.y + obs.radius < minY || obs.y - obs.radius > maxY) {
            continue;
          }
        } else if (obs.type === 'rect') {
          if (obs.x + obs.width < minX || obs.x > maxX ||
              obs.y + obs.height < minY || obs.y > maxY) {
            continue;
          }
        }
        
        const intersection = obs.intersectRay(startPoint, endPoint);
        if (intersection && intersection.t < minT) {
          minT = intersection.t;
          closestPoint = intersection;
        }
      }
      
      this.sensorReadings[i] = minT;
      this.sensorPoints[i] = closestPoint;
    }
  }

  checkCollisions(obstacles, boundaryWidth, boundaryHeight) {
    if (this.x - this.radius < 0 || this.x + this.radius > boundaryWidth ||
        this.y - this.radius < 0 || this.y + this.radius > boundaryHeight) {
      this.isDead = true;
      this.calculateFitness();
      return;
    }
    
    for (let obs of obstacles) {
      if (obs.collidesWithCircle(this.x, this.y, this.radius)) {
        this.isDead = true;
        this.calculateFitness();
        return;
      }
    }
  }

  calculateFitness() {
    if (this.hasSucceeded) {
      this.fitness = 3000 + Math.max(0, 1000 - this.stepsTaken);
    } else {
      const progress = 1.0 - (this.closestDist / this.startDist);
      this.fitness = Math.max(0.1, progress * 1000);
      this.fitness *= 0.65;
    }
  }

  draw(ctx, showSensors, isBest, theme) {
    ctx.save();
    
    // 1. Draw sensors
    if (showSensors && !this.isDead && !this.hasSucceeded) {
      for (let i = 0; i < this.sensorPoints.length; i++) {
        const pt = this.sensorPoints[i];
        if (!pt) continue;
        
        const isClose = this.sensorReadings[i] < 0.35;
        ctx.strokeStyle = isClose ? 'rgba(239, 68, 68, 0.25)' : 'rgba(6, 182, 212, 0.12)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(pt.x, pt.y);
        ctx.stroke();
        
        if (this.sensorReadings[i] < 1.0) {
          ctx.fillStyle = isClose ? '#ef4444' : '#06b6d4';
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // 2. Draw flight/crawling trail with dynamic colors
    if (this.trail.length > 1) {
      if (isBest) {
        ctx.strokeStyle = theme === 'ants' ? 'rgba(168, 85, 247, 0.3)' 
                        : theme === 'drones' ? 'rgba(6, 182, 212, 0.3)' 
                        : theme === 'beetles' ? 'rgba(239, 68, 68, 0.3)' 
                        : theme === 'bacteria' ? 'rgba(34, 197, 94, 0.3)' 
                        : 'rgba(245, 158, 11, 0.3)';
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = theme === 'ants' ? 'rgba(168, 85, 247, 0.12)' 
                        : theme === 'drones' ? 'rgba(6, 182, 212, 0.12)' 
                        : theme === 'beetles' ? 'rgba(239, 68, 68, 0.12)' 
                        : theme === 'bacteria' ? 'rgba(34, 197, 94, 0.12)' 
                        : 'rgba(99, 102, 241, 0.1)';
        ctx.lineWidth = 1.5;
      }
      ctx.beginPath();
      ctx.moveTo(this.trail[0].x, this.trail[0].y);
      for (let j = 1; j < this.trail.length; j++) {
        ctx.lineTo(this.trail[j].x, this.trail[j].y);
      }
      ctx.stroke();
    }
    
    // 3. Highlight Aura (if best)
    if (isBest && !this.isDead) {
      ctx.shadowBlur = 15;
      ctx.shadowColor = theme === 'ants' ? 'rgba(168, 85, 247, 0.8)' 
                      : theme === 'drones' ? 'rgba(6, 182, 212, 0.8)' 
                      : theme === 'beetles' ? 'rgba(239, 68, 68, 0.8)' 
                      : theme === 'bacteria' ? 'rgba(34, 197, 94, 0.8)' 
                      : 'rgba(245, 158, 11, 0.8)';
      ctx.strokeStyle = theme === 'ants' ? '#a855f7' 
                      : theme === 'drones' ? '#06b6d4' 
                      : theme === 'beetles' ? '#ef4444' 
                      : theme === 'bacteria' ? '#22c55e' 
                      : '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius + 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
    
    // Death cross
    if (this.isDead) {
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(this.x - 5, this.y - 5);
      ctx.lineTo(this.x + 5, this.y + 5);
      ctx.moveTo(this.x + 5, this.y - 5);
      ctx.lineTo(this.x - 5, this.y + 5);
      ctx.stroke();
      ctx.restore();
      return;
    }
    
    // Success ring
    if (this.hasSucceeded) {
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
      ctx.shadowColor = 'rgba(16, 185, 129, 0.6)';
      ctx.shadowBlur = 10;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius + 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      return;
    }
    
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);
    
    // Draw vector designs based on theme
    if (theme === 'ants') {
      // --- ANT DRAWING ---
      ctx.fillStyle = '#334155'; // Dark metallic slate
      
      // Abdomen
      ctx.beginPath();
      ctx.ellipse(-5, 0, 6, 3.5, 0, 0, Math.PI * 2);
      ctx.fill();
      
      // Thorax
      ctx.beginPath();
      ctx.arc(0, 0, 3, 0, Math.PI * 2);
      ctx.fill();
      
      // Head
      ctx.beginPath();
      ctx.arc(5, 0, 2.8, 0, Math.PI * 2);
      ctx.fill();
      
      // 6 crawling legs (swing based on phase)
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 1;
      const legSwing = Math.sin(this.wingPhase) * 0.25;
      
      ctx.beginPath();
      // Left 3 legs
      ctx.moveTo(0, -1.5); ctx.lineTo(-2 + legSwing, -7);
      ctx.moveTo(0, -1.5); ctx.lineTo(legSwing, -8);
      ctx.moveTo(0, -1.5); ctx.lineTo(2 + legSwing, -7);
      // Right 3 legs
      ctx.moveTo(0, 1.5); ctx.lineTo(-2 - legSwing, 7);
      ctx.moveTo(0, 1.5); ctx.lineTo(-legSwing, 8);
      ctx.moveTo(0, 1.5); ctx.lineTo(2 - legSwing, 7);
      ctx.stroke();
      
      // Antennae
      ctx.strokeStyle = '#64748b';
      ctx.beginPath();
      ctx.moveTo(6.5, -1.5); ctx.lineTo(9.5, -3.5);
      ctx.moveTo(6.5, 1.5); ctx.lineTo(9.5, 3.5);
      ctx.stroke();
      
      // Glowing purple eyes
      ctx.fillStyle = '#c084fc';
      ctx.beginPath();
      ctx.arc(6.8, -1.2, 0.7, 0, Math.PI * 2);
      ctx.arc(6.8, 1.2, 0.7, 0, Math.PI * 2);
      ctx.fill();
      
    } else if (theme === 'drones') {
      // --- DRONE DRAWING ---
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 1.2;
      
      // X configuration arm frame
      ctx.beginPath();
      ctx.moveTo(-6, -6); ctx.lineTo(6, 6);
      ctx.moveTo(-6, 6); ctx.lineTo(6, -6);
      ctx.stroke();
      
      // Propeller centers
      ctx.fillStyle = '#1e293b';
      ctx.beginPath();
      ctx.arc(-6, -6, 1.5, 0, Math.PI * 2);
      ctx.arc(6, -6, 1.5, 0, Math.PI * 2);
      ctx.arc(-6, 6, 1.5, 0, Math.PI * 2);
      ctx.arc(6, 6, 1.5, 0, Math.PI * 2);
      ctx.fill();
      
      // Spin blades
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 0.8;
      const propAngle = this.wingPhase * 2.5;
      
      const drawBlade = (px, py) => {
        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(propAngle);
        ctx.beginPath();
        ctx.moveTo(-4.5, 0); ctx.lineTo(4.5, 0);
        ctx.stroke();
        ctx.restore();
      };
      
      drawBlade(-6, -6);
      drawBlade(6, -6);
      drawBlade(-6, 6);
      drawBlade(6, 6);
      
      // Central Pod Body
      ctx.fillStyle = '#0f172a';
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(0, 0, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      // Front camera glow (cyan)
      ctx.fillStyle = '#06b6d4';
      ctx.beginPath();
      ctx.arc(3, 0, 1.8, -Math.PI / 2, Math.PI / 2);
      ctx.fill();
      
    } else if (theme === 'beetles') {
      // --- BEETLE DRAWING ---
      ctx.fillStyle = '#ef4444'; // Red carapace
      ctx.beginPath();
      ctx.ellipse(0, 0, 9, 8, 0, 0, Math.PI * 2);
      ctx.fill();
      
      // Center split line
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(-9, 0); ctx.lineTo(9, 0);
      ctx.stroke();
      
      // Spots
      ctx.fillStyle = '#0f172a';
      ctx.beginPath();
      ctx.arc(-3, -3, 1.5, 0, Math.PI * 2);
      ctx.arc(-3, 3, 1.5, 0, Math.PI * 2);
      ctx.arc(3, -3, 1.5, 0, Math.PI * 2);
      ctx.arc(3, 3, 1.5, 0, Math.PI * 2);
      ctx.fill();
      
      // Head
      ctx.fillStyle = '#334155';
      ctx.beginPath();
      ctx.arc(7, 0, 3, 0, Math.PI * 2);
      ctx.fill();
      
      // Crawling legs
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 1;
      const legSwing = Math.sin(this.wingPhase) * 0.25;
      ctx.beginPath();
      ctx.moveTo(2, -3); ctx.lineTo(4 + legSwing, -8);
      ctx.moveTo(-1, -3.5); ctx.lineTo(-1 + legSwing, -9);
      ctx.moveTo(-4, -3); ctx.lineTo(-6 + legSwing, -8);
      ctx.moveTo(2, 3); ctx.lineTo(4 - legSwing, 8);
      ctx.moveTo(-1, 3.5); ctx.lineTo(-1 - legSwing, 9);
      ctx.moveTo(-4, 3); ctx.lineTo(-6 - legSwing, 8);
      ctx.stroke();
      
    } else if (theme === 'bacteria') {
      // --- BACTERIA DRAWING ---
      ctx.fillStyle = 'rgba(34, 197, 94, 0.65)'; // Translucent green
      ctx.strokeStyle = '#22c55e'; // Green membrane
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(0, 0, 10, 5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      // Wiggling tails (sine wave flagella)
      ctx.strokeStyle = '#4ade80';
      ctx.lineWidth = 1;
      const wavePhase = this.wingPhase * 1.5;
      
      ctx.beginPath();
      ctx.moveTo(-10, -2);
      for (let x = 0; x < 15; x++) {
        const yOffset = Math.sin(x * 0.4 - wavePhase) * 2.5;
        ctx.lineTo(-10 - x, -2 + yOffset);
      }
      ctx.moveTo(-10, 2);
      for (let x = 0; x < 15; x++) {
        const yOffset = Math.sin(x * 0.4 - wavePhase + Math.PI) * 2.5;
        ctx.lineTo(-10 - x, 2 + yOffset);
      }
      ctx.stroke();
      
      // internal organelles
      ctx.fillStyle = '#bbf7d0';
      ctx.beginPath();
      ctx.arc(-2, -1.5, 1.2, 0, Math.PI * 2);
      ctx.arc(3, 1, 1, 0, Math.PI * 2);
      ctx.fill();
      
    } else {
      // --- BEE DRAWING (Default) ---
      ctx.fillStyle = 'rgba(200, 230, 255, 0.65)';
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.lineWidth = 1;
      
      const wingSizeX = 11;
      const wingSizeY = 5;
      const wingSpread = Math.sin(this.wingPhase) * 0.4;
      
      ctx.save();
      ctx.translate(0, -2);
      ctx.rotate(-Math.PI / 3 + wingSpread);
      ctx.beginPath();
      ctx.ellipse(0, -wingSizeX/2, wingSizeY, wingSizeX, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      
      ctx.save();
      ctx.translate(0, 2);
      ctx.rotate(Math.PI / 3 - wingSpread);
      ctx.beginPath();
      ctx.ellipse(0, wingSizeX/2, wingSizeY, wingSizeX, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      
      ctx.fillStyle = '#1e293b';
      ctx.beginPath();
      ctx.ellipse(0, 0, 11, 7, 0, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.fillStyle = '#f59e0b';
      ctx.fillRect(-4, -6.5, 3, 13);
      ctx.fillRect(2, -6, 3, 12);
      
      ctx.fillStyle = '#06b6d4';
      ctx.beginPath();
      ctx.arc(8, 0, 4, -Math.PI / 2, Math.PI / 2);
      ctx.fill();
    }
    
    ctx.restore();
  }
}

// ==========================================
// SECTION 5: GENETIC ALGORITHM ENGINE
// ==========================================

function nextGeneration(
  currentBees,
  popSize,
  mutationRate,
  mutationAmount,
  elitismPct,
  selectionMethod,
  startX,
  startY,
  brainTopology
) {
  const sortedBees = [...currentBees].sort((a, b) => b.fitness - a.fitness);
  const nextPop = [];

  const eliteCount = Math.floor((popSize * elitismPct) / 100);
  
  for (let i = 0; i < eliteCount; i++) {
    if (sortedBees[i]) {
      const eliteBee = new Bee(startX, startY, brainTopology, sortedBees[i].brain);
      nextPop.push(eliteBee);
    }
  }

  const reversedBees = [...sortedBees].reverse();
  const totalRanks = (popSize * (popSize + 1)) / 2;

  while (nextPop.length < popSize) {
    const parentA = selectParent(sortedBees, reversedBees, totalRanks, selectionMethod);
    const parentB = selectParent(sortedBees, reversedBees, totalRanks, selectionMethod);
    
    let childBrain = parentA.brain.crossover(parentB.brain);
    childBrain.mutate(mutationRate, mutationAmount);
    
    const childBee = new Bee(startX, startY, brainTopology, childBrain);
    nextPop.push(childBee);
  }

  return nextPop;
}

function selectParent(sortedBees, reversedBees, totalRanks, method) {
  const n = sortedBees.length;
  
  if (method === 'tournament') {
    const tournamentSize = 4;
    let best = null;
    for (let i = 0; i < tournamentSize; i++) {
      const index = Math.floor(Math.random() * n);
      const contender = sortedBees[index];
      if (!best || contender.fitness > best.fitness) {
        best = contender;
      }
    }
    return best;
    
  } else if (method === 'roulette') {
    let totalFitness = sortedBees.reduce((sum, bee) => sum + bee.fitness, 0);
    
    if (totalFitness <= 0) {
      return sortedBees[Math.floor(Math.random() * n)];
    }
    
    const target = Math.random() * totalFitness;
    let currentSum = 0;
    
    for (let bee of sortedBees) {
      currentSum += bee.fitness;
      if (currentSum >= target) return bee;
    }
    return sortedBees[n - 1];
    
  } else if (method === 'rank') {
    const target = Math.random() * totalRanks;
    let currentSum = 0;
    
    for (let i = 0; i < n; i++) {
      const rank = i + 1;
      currentSum += rank;
      if (currentSum >= target) return reversedBees[i];
    }
    return reversedBees[n - 1];
  }
  
  return sortedBees[Math.floor(Math.random() * n)];
}

// ==========================================
// SECTION 6: CANVAS VISUALIZATION DRAWERS
// ==========================================

function drawNeuralNetwork(canvas, activations, layers) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  
  ctx.clearRect(0, 0, width, height);
  
  if (!activations || !layers || activations.length === 0) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.font = '13px var(--font-sans)';
    ctx.textAlign = 'center';
    ctx.fillText('Select a bee to view brain activity', width / 2, height / 2);
    return;
  }

  const layerSizes = activations.map(arr => arr.length);
  const layerCount = layerSizes.length;
  
  const paddingX = 45;
  const paddingY = 20;
  const layerSpacingX = (width - paddingX * 2) / (layerCount - 1);
  
  const nodes = [];
  for (let i = 0; i < layerCount; i++) {
    const size = layerSizes[i];
    const layerNodes = [];
    const layerX = paddingX + i * layerSpacingX;
    
    for (let j = 0; j < size; j++) {
      let layerY;
      if (size === 1) {
        layerY = height / 2;
      } else {
        layerY = paddingY + j * ((height - paddingY * 2) / (size - 1));
      }
      layerNodes.push({ x: layerX, y: layerY });
    }
    nodes.push(layerNodes);
  }

  for (let l = 0; l < layers.length; l++) {
    const currentLayer = layers[l];
    const prevNodes = nodes[l];
    const nextNodes = nodes[l + 1];
    
    for (let i = 0; i < currentLayer.outputSize; i++) {
      const nextNode = nextNodes[i];
      for (let j = 0; j < currentLayer.inputSize; j++) {
        const prevNode = prevNodes[j];
        const weight = currentLayer.weights[i][j];
        
        ctx.lineWidth = Math.min(Math.abs(weight) * 1.5, 3);
        if (ctx.lineWidth < 0.1) continue;
        
        ctx.strokeStyle = weight > 0 
          ? `rgba(6, 182, 212, ${Math.min(Math.abs(weight) * 0.3, 0.55)})` 
          : `rgba(244, 63, 94, ${Math.min(Math.abs(weight) * 0.3, 0.55)})`;
          
        ctx.beginPath();
        ctx.moveTo(prevNode.x, prevNode.y);
        ctx.lineTo(nextNode.x, nextNode.y);
        ctx.stroke();
      }
    }
  }

  const animTime = (Date.now() / 1500) % 1.0;
  for (let l = 0; l < layers.length; l++) {
    const currentLayer = layers[l];
    const prevNodes = nodes[l];
    const nextNodes = nodes[l + 1];
    
    for (let i = 0; i < currentLayer.outputSize; i++) {
      const nextNode = nextNodes[i];
      for (let j = 0; j < currentLayer.inputSize; j++) {
        const prevNode = prevNodes[j];
        const weight = currentLayer.weights[i][j];
        const prevActivation = activations[l][j];
        
        if (Math.abs(prevActivation) > 0.1 && Math.abs(weight) > 0.2) {
          ctx.fillStyle = weight > 0 ? '#06b6d4' : '#f43f5e';
          const px = prevNode.x + (nextNode.x - prevNode.x) * animTime;
          const py = prevNode.y + (nextNode.y - prevNode.y) * animTime;
          ctx.beginPath();
          ctx.arc(px, py, 1.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  const inputLabels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'T_dx', 'T_dy', 'V_vx', 'V_vy'];
  const outputLabels = ['Steer', 'Thrust'];
  
  for (let l = 0; l < layerCount; l++) {
    const size = layerSizes[l];
    for (let i = 0; i < size; i++) {
      const node = nodes[l][i];
      const activation = activations[l][i];
      
      ctx.save();
      const nodeRadius = 5.5;
      ctx.shadowBlur = Math.abs(activation) * 8;
      
      if (activation > 0.05) {
        ctx.fillStyle = `rgba(245, 158, 11, ${0.4 + activation * 0.6})`;
        ctx.shadowColor = '#f59e0b';
      } else if (activation < -0.05) {
        ctx.fillStyle = `rgba(6, 182, 212, ${0.4 + Math.abs(activation) * 0.6})`;
        ctx.shadowColor = '#06b6d4';
      } else {
        ctx.fillStyle = '#334155';
        ctx.shadowBlur = 0;
      }
      
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      
      ctx.beginPath();
      ctx.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      
      if (l === 0) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.font = '8px var(--font-mono)';
        ctx.textAlign = 'right';
        ctx.fillText(inputLabels[i] || `In${i}`, node.x - 8, node.y + 3);
      } else if (l === layerCount - 1) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.font = '9px var(--font-sans)';
        ctx.textAlign = 'left';
        ctx.fillText(outputLabels[i] || `Out${i}`, node.x + 8, node.y + 3);
      }
    }
  }
}

function drawProgressChart(canvas, history) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  
  ctx.clearRect(0, 0, width, height);
  
  if (history.length === 0) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.font = '13px var(--font-sans)';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for first generation to complete...', width / 2, height / 2);
    return;
  }

  const paddingLeft = 40;
  const paddingRight = 15;
  const paddingTop = 15;
  const paddingBottom = 25;
  
  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;

  let maxFitness = 100;
  for (let record of history) {
    if (record.bestFitness > maxFitness) maxFitness = record.bestFitness;
    if (record.avgFitness > maxFitness) maxFitness = record.avgFitness;
  }
  maxFitness *= 1.1;

  const maxGen = Math.max(1, history[history.length - 1].generation);
  const minGen = 1;
  const genSpan = maxGen - minGen;

  const getX = (gen) => {
    if (genSpan === 0) return paddingLeft + chartWidth / 2;
    return paddingLeft + ((gen - minGen) / genSpan) * chartWidth;
  };
  const getY = (fitness) => {
    return paddingTop + chartHeight - (fitness / maxFitness) * chartHeight;
  };

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  
  ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.font = '9px var(--font-mono)';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  
  for (let i = 0; i <= 4; i++) {
    const val = (maxFitness / 4) * i;
    const y = getY(val);
    ctx.beginPath();
    ctx.moveTo(paddingLeft, y);
    ctx.lineTo(width - paddingRight, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(0), paddingLeft - 6, y);
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  
  const gridStep = Math.max(1, Math.ceil(maxGen / 5));
  for (let g = 1; g <= maxGen; g += gridStep) {
    const x = getX(g);
    ctx.beginPath();
    ctx.moveTo(x, paddingTop);
    ctx.lineTo(x, paddingTop + chartHeight);
    ctx.stroke();
    ctx.fillText(g.toString(), x, paddingTop + chartHeight + 6);
  }
  
  ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
  ctx.beginPath();
  ctx.moveTo(paddingLeft, paddingTop);
  ctx.lineTo(paddingLeft, paddingTop + chartHeight);
  ctx.lineTo(width - paddingRight, paddingTop + chartHeight);
  ctx.stroke();

  ctx.save();
  ctx.strokeStyle = '#6366f1';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(getX(history[0].generation), getY(history[0].avgFitness));
  for (let j = 1; j < history.length; j++) {
    ctx.lineTo(getX(history[j].generation), getY(history[j].avgFitness));
  }
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2.5;
  ctx.shadowColor = 'rgba(245, 158, 11, 0.4)';
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(getX(history[0].generation), getY(history[0].bestFitness));
  for (let j = 1; j < history.length; j++) {
    ctx.lineTo(getX(history[j].generation), getY(history[j].bestFitness));
  }
  ctx.stroke();
  ctx.restore();

  ctx.font = '9px var(--font-sans)';
  ctx.textAlign = 'left';
  
  ctx.fillStyle = '#f59e0b';
  ctx.beginPath();
  ctx.arc(paddingLeft + 10, paddingTop + 8, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.fillText('Best Fitness', paddingLeft + 18, paddingTop + 8);
  
  ctx.fillStyle = '#6366f1';
  ctx.beginPath();
  ctx.arc(paddingLeft + 100, paddingTop + 8, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.fillText('Avg Fitness', paddingLeft + 108, paddingTop + 8);
}

// ==========================================
// SECTION 7: SIMULATION ORCHESTRATOR
// ==========================================

class Simulation {
  constructor(canvas, onGenCompleteCallback, onStepCallback) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    
    this.bees = [];
    this.obstacles = [];
    this.target = { x: 0, y: 0, radius: 12 };
    
    this.startX = 0;
    this.startY = 0;
    
    this.currentStep = 0;
    this.maxGenSteps = 900;
    this.generation = 1;
    
    this.isPaused = true;
    this.speed = 1;
    this.theme = 'bees'; // 'bees', 'ants', 'drones'
    
    this.showSensors = true;
    this.showBest = true;
    this.showTrails = true;
    
    this.onGenerationComplete = onGenCompleteCallback;
    this.onStep = onStepCallback;
    
    this.activeCount = 0;
    this.bestBee = null;
    this.successCount = 0;
    
    this.init();
  }

  init() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
    
    this.startX = this.canvas.width / 2;
    this.startY = this.canvas.height - 50;
    
    this.target.x = this.canvas.width / 2;
    this.target.y = 70;
  }

  startGeneration(population) {
    this.bees = population;
    this.currentStep = 0;
    this.activeCount = population.length;
    this.successCount = 0;
    this.bestBee = population[0];
    
    for (let bee of this.bees) {
      bee.x = this.startX;
      bee.y = this.startY;
      bee.vx = 0;
      bee.vy = 0;
      bee.angle = -Math.PI / 2;
      bee.isDead = false;
      bee.hasSucceeded = false;
      bee.trail = [];
      bee.stepsTaken = 0;
      bee.lastActivations = null;
      bee.closestDist = Math.hypot(this.target.x - bee.x, this.target.y - bee.y);
      bee.startDist = bee.closestDist;
    }
  }

  update() {
    if (this.isPaused || this.bees.length === 0) return;

    for (let s = 0; s < this.speed; s++) {
      this.currentStep++;
      this.activeCount = 0;
      this.successCount = 0;
      
      for (let obs of this.obstacles) {
        obs.update();
      }
      
      let currentBest = null;

      for (let bee of this.bees) {
        bee.update(this.obstacles, this.target, this.canvas.width, this.canvas.height, this.currentStep);
        
        if (!bee.isDead && !bee.hasSucceeded) {
          this.activeCount++;
        }
        if (bee.hasSucceeded) {
          this.successCount++;
        }
        
        bee.calculateFitness();
        if (!currentBest || bee.fitness > currentBest.fitness) {
          currentBest = bee;
        }
      }
      
      if (currentBest) {
        this.bestBee = currentBest;
      }
      
      if (this.onStep) {
        this.onStep();
      }

      if (this.activeCount === 0 || this.currentStep >= this.maxGenSteps) {
        for (let bee of this.bees) {
          bee.calculateFitness();
        }
        this.isPaused = true;
        
        if (this.onGenerationComplete) {
          this.onGenerationComplete();
        }
        break;
      }
    }
  }

  draw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // 1. Draw Target (Themed)
    this.ctx.save();
    const pulse = 1.0 + Math.sin(Date.now() / 150) * 0.08;
    
    if (this.theme === 'ants') {
      // Glowing sugar crystal (diamond-shape)
      this.ctx.shadowBlur = 18 * pulse;
      this.ctx.shadowColor = '#a855f7';
      this.ctx.fillStyle = '#c084fc';
      this.ctx.beginPath();
      this.ctx.moveTo(this.target.x, this.target.y - 12 * pulse);
      this.ctx.lineTo(this.target.x + 8 * pulse, this.target.y);
      this.ctx.lineTo(this.target.x, this.target.y + 12 * pulse);
      this.ctx.lineTo(this.target.x - 8 * pulse, this.target.y);
      this.ctx.closePath();
      this.ctx.fill();
    } else if (this.theme === 'drones') {
      // Glowing tech portal orb (cyan)
      this.ctx.shadowBlur = 20 * pulse;
      this.ctx.shadowColor = 'rgba(6, 182, 212, 0.8)';
      this.ctx.fillStyle = '#06b6d4';
      this.ctx.beginPath();
      this.ctx.arc(this.target.x, this.target.y, this.target.radius * pulse, 0, Math.PI * 2);
      this.ctx.fill();
      
      this.ctx.shadowBlur = 0;
      this.ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      this.ctx.arc(this.target.x, this.target.y, this.target.radius + 8, 0, Math.PI * 2);
      this.ctx.stroke();
    } else if (this.theme === 'beetles') {
      // Glowing green leaf
      this.ctx.shadowBlur = 18 * pulse;
      this.ctx.shadowColor = '#22c55e';
      this.ctx.fillStyle = '#22c55e';
      this.ctx.beginPath();
      this.ctx.ellipse(this.target.x, this.target.y, 10 * pulse, 6 * pulse, Math.PI / 4, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.strokeStyle = '#16803d';
      this.ctx.lineWidth = 1;
      this.ctx.beginPath();
      this.ctx.moveTo(this.target.x - 6, this.target.y + 4);
      this.ctx.lineTo(this.target.x + 6, this.target.y - 4);
      this.ctx.stroke();
    } else if (this.theme === 'bacteria') {
      // Glowing cyan nutrient droplet
      this.ctx.shadowBlur = 20 * pulse;
      this.ctx.shadowColor = '#06b6d4';
      this.ctx.fillStyle = '#22d3ee';
      this.ctx.beginPath();
      this.ctx.arc(this.target.x, this.target.y, 8 * pulse, 0, Math.PI * 2);
      this.ctx.fill();
    } else {
      // Cyber-Bees: glowing center with outer petals (glowing flower)
      this.ctx.shadowBlur = 18 * pulse;
      this.ctx.shadowColor = 'rgba(244, 63, 94, 0.8)';
      this.ctx.fillStyle = 'rgba(244, 63, 94, 0.65)';
      for (let i = 0; i < 5; i++) {
        const angle = (i * Math.PI * 2) / 5 + (Date.now() / 1500);
        const px = this.target.x + Math.cos(angle) * (8 * pulse);
        const py = this.target.y + Math.sin(angle) * (8 * pulse);
        this.ctx.beginPath();
        this.ctx.arc(px, py, 5, 0, Math.PI * 2);
        this.ctx.fill();
      }
      this.ctx.fillStyle = '#fbbf24';
      this.ctx.beginPath();
      this.ctx.arc(this.target.x, this.target.y, 5 * pulse, 0, Math.PI * 2);
      this.ctx.fill();
    }
    
    // Rotating tech ring indicators
    this.ctx.restore();
    this.ctx.save();
    const ringColor = this.theme === 'ants' ? 'rgba(168, 85, 247, 0.3)' 
                    : this.theme === 'drones' ? 'rgba(6, 182, 212, 0.3)' 
                    : this.theme === 'beetles' ? 'rgba(34, 197, 94, 0.3)'
                    : this.theme === 'bacteria' ? 'rgba(6, 182, 212, 0.3)'
                    : 'rgba(16, 185, 129, 0.3)';
    this.ctx.strokeStyle = ringColor;
    this.ctx.lineWidth = 1;
    this.ctx.setLineDash([4, 4]);
    this.ctx.translate(this.target.x, this.target.y);
    this.ctx.rotate(Date.now() / 800);
    this.ctx.beginPath();
    this.ctx.arc(0, 0, this.target.radius + 13, 0, Math.PI * 2);
    this.ctx.stroke();
    this.ctx.restore();

    // 2. Draw Start (Themed launch zone)
    this.ctx.save();
    if (this.theme === 'ants') {
      // Dirt brown Anthill mound
      this.ctx.fillStyle = 'rgba(120, 113, 108, 0.2)';
      this.ctx.strokeStyle = 'rgba(120, 113, 108, 0.5)';
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      this.ctx.arc(this.startX, this.startY, 16, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.stroke();
      
      this.ctx.fillStyle = '#1c1917';
      this.ctx.beginPath();
      this.ctx.arc(this.startX, this.startY, 4, 0, Math.PI * 2);
      this.ctx.fill();
      
      this.ctx.fillStyle = 'rgba(120, 113, 108, 0.4)';
      this.ctx.font = '700 8px var(--font-sans)';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText('NEST', this.startX, this.startY - 11);
      
    } else if (this.theme === 'drones') {
      // Futuristic landing pad grid
      this.ctx.strokeStyle = 'rgba(6, 182, 212, 0.3)';
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      this.ctx.arc(this.startX, this.startY, 16, 0, Math.PI * 2);
      this.ctx.stroke();
      
      this.ctx.strokeStyle = 'rgba(6, 182, 212, 0.15)';
      this.ctx.beginPath();
      this.ctx.arc(this.startX, this.startY, 22, 0, Math.PI * 2);
      this.ctx.moveTo(this.startX - 26, this.startY);
      this.ctx.lineTo(this.startX + 26, this.startY);
      this.ctx.moveTo(this.startX, this.startY - 26);
      this.ctx.lineTo(this.startX, this.startY + 26);
      this.ctx.stroke();
      
      this.ctx.fillStyle = 'rgba(6, 182, 212, 0.4)';
      this.ctx.font = '700 9px var(--font-sans)';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText('PAD', this.startX, this.startY);
      
    } else if (this.theme === 'beetles') {
      // Twig/Leaf Nest
      this.ctx.strokeStyle = 'rgba(34, 197, 94, 0.4)';
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      this.ctx.ellipse(this.startX, this.startY, 20, 10, 0, 0, Math.PI * 2);
      this.ctx.stroke();
      
      this.ctx.fillStyle = 'rgba(34, 197, 94, 0.4)';
      this.ctx.font = '700 9px var(--font-sans)';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText('LEAF', this.startX, this.startY);
      
    } else if (this.theme === 'bacteria') {
      // Circular Petri dish
      this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      this.ctx.arc(this.startX, this.startY, 16, 0, Math.PI * 2);
      this.ctx.stroke();
      
      this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.15)';
      this.ctx.beginPath();
      this.ctx.arc(this.startX, this.startY, 22, 0, Math.PI * 2);
      this.ctx.stroke();
      
      this.ctx.fillStyle = 'rgba(16, 185, 129, 0.4)';
      this.ctx.font = '700 9px var(--font-sans)';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText('PLATE', this.startX, this.startY);
      
    } else {
      // Bees Hexagonal Honeycomb hangar
      this.ctx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
      this.ctx.lineWidth = 1.2;
      this.ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = (i * Math.PI) / 3;
        const hx = this.startX + Math.cos(angle) * 16;
        const hy = this.startY + Math.sin(angle) * 16;
        if (i === 0) this.ctx.moveTo(hx, hy);
        else this.ctx.lineTo(hx, hy);
      }
      this.ctx.closePath();
      this.ctx.stroke();
      
      this.ctx.fillStyle = 'rgba(245, 158, 11, 0.4)';
      this.ctx.font = '700 9px var(--font-sans)';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText('HIVE', this.startX, this.startY);
    }
    this.ctx.restore();

    // 3. Draw Obstacles
    for (let obs of this.obstacles) {
      obs.draw(this.ctx);
    }

    // 4. Draw Drones/Agents
    const sortedBeesForDraw = [...this.bees].sort((a, b) => {
      if (a === this.bestBee) return 1;
      if (b === this.bestBee) return -1;
      if (a.isDead && !b.isDead) return -1;
      if (!a.isDead && b.isDead) return 1;
      if (a.hasSucceeded && !b.hasSucceeded) return -1;
      if (!a.hasSucceeded && b.hasSucceeded) return 1;
      return 0;
    });

    for (let bee of sortedBeesForDraw) {
      const isBest = bee === this.bestBee && this.showBest;
      bee.draw(this.ctx, this.showSensors, isBest, this.theme);
    }
  }

  loadPresetMap(name) {
    this.obstacles = [];
    const w = this.canvas.width;
    const h = this.canvas.height;
    
    this.target.x = w / 2;
    this.target.y = 70;
    this.startX = w / 2;
    this.startY = h - 50;

    switch (name) {
      case 'simple-maze': {
        this.obstacles.push(new RectObstacle(w * 0.15, h * 0.45, w * 0.3, 30));
        this.obstacles.push(new RectObstacle(w * 0.55, h * 0.45, w * 0.3, 30));
        this.obstacles.push(new CircleObstacle(w / 2, h * 0.72, 35));
        break;
      }
      case 'zigzag': {
        this.obstacles.push(new RectObstacle(0, h * 0.3, w * 0.7, 25));
        this.obstacles.push(new RectObstacle(w * 0.3, h * 0.52, w * 0.7, 25));
        this.obstacles.push(new RectObstacle(0, h * 0.72, w * 0.6, 25));
        break;
      }
      case 'dynamic-gate': {
        const gate1 = new RectObstacle(w * 0.3, h * 0.38, w * 0.4, 25);
        gate1.setMotion(2.5, 0, w * 0.05, w * 0.55, null, null);
        
        const gate2 = new RectObstacle(w * 0.1, h * 0.62, w * 0.35, 25);
        gate2.setMotion(-3, 0, w * 0.05, w * 0.6, null, null);
        
        this.obstacles.push(gate1);
        this.obstacles.push(gate2);
        this.obstacles.push(new CircleObstacle(w / 2, h * 0.5, 20));
        break;
      }
      case 'ring-fire': {
        const cx = w / 2;
        const cy = h / 2 - 20;
        const count = 6;
        const dist = 90;
        
        for (let i = 0; i < count; i++) {
          const angle = (i * Math.PI * 2) / count;
          const ox = cx + Math.cos(angle) * dist;
          const oy = cy + Math.sin(angle) * dist;
          const circle = new CircleObstacle(ox, oy, 22);
          if (i === 0 || i === 3) {
            circle.setMotion(0, 1.2, null, null, oy - 25, oy + 25);
          }
          this.obstacles.push(circle);
        }
        this.obstacles.push(new CircleObstacle(cx, cy, 18));
        break;
      }
      case 'empty':
      default:
        break;
    }
  }
}

// ==========================================
// SECTION 8: DOM ORCHESTRATION & APP MAIN
// ==========================================

let population = [];
let history = [];
let generation = 1;
let isSimulationRunning = false;
let currentTheme = 'bees'; // 'bees', 'ants', 'drones'

let isDrawing = false;
let drawStart = { x: 0, y: 0 };
let drawCurrent = { x: 0, y: 0 };
let activeTool = 'brush';
let lastBrushPos = null;

const canvasSim = document.getElementById('canvas-sim');
const canvasNN = document.getElementById('canvas-nn-viz');
const canvasChart = document.getElementById('canvas-chart-viz');
const overlay = document.getElementById('canvas-overlay');

const btnPlayPause = document.getElementById('btn-play-pause');
const btnNextGen = document.getElementById('btn-next-gen');
const btnReset = document.getElementById('btn-reset');
const txtPlayPause = document.getElementById('txt-play-pause');
const btnClearObstacles = document.getElementById('btn-clear-obstacles');

const sliderSpeed = document.getElementById('slider-speed');
const valSpeed = document.getElementById('val-speed');
const selectPreset = document.getElementById('preset-map');
const selectTheme = document.getElementById('select-theme');
const sliderBrushSize = document.getElementById('slider-brush-size');
const valBrushSize = document.getElementById('val-brush-size');

const inputPopSize = document.getElementById('input-pop-size');
const valPopSize = document.getElementById('val-pop-size');
const inputMutationRate = document.getElementById('input-mutation-rate');
const valMutationRate = document.getElementById('val-mutation-rate');
const inputMutationAmount = document.getElementById('input-mutation-amount');
const valMutationAmount = document.getElementById('val-mutation-amount');
const inputElitePct = document.getElementById('input-elite-pct');
const valElitePct = document.getElementById('val-elite-pct');
const selectSelection = document.getElementById('select-selection');
const inputHiddenLayers = document.getElementById('input-hidden-layers');

const textGeneration = document.getElementById('metric-gen');
const textHeaderGen = document.getElementById('header-generation');
const textSuccess = document.getElementById('metric-success');
const textHeaderSuccess = document.getElementById('header-success-rate');
const textBest = document.getElementById('metric-best');
const textHeaderBest = document.getElementById('header-best-fitness');
const textAvg = document.getElementById('metric-avg');
const textActiveCount = document.getElementById('txt-active-count');
const progressBarGenTime = document.getElementById('progress-gen-time');
const textTime = document.getElementById('metric-time');
const visualizedBeeId = document.getElementById('visualized-bee-id');

const chkShowSensors = document.getElementById('chk-show-sensors');
const chkShowBest = document.getElementById('chk-show-best');
const chkShowTrails = document.getElementById('chk-show-trails');

const toolButtons = document.querySelectorAll('.btn-tool');

function getMousePos(canvas, evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((evt.clientX - rect.left) / (rect.right - rect.left)) * canvas.width,
    y: ((evt.clientY - rect.top) / (rect.bottom - rect.top)) * canvas.height
  };
}

function parseHiddenLayers() {
  const str = inputHiddenLayers.value.trim();
  if (!str) return [8, 6];
  const layers = str.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0);
  return layers.length > 0 ? layers : [8, 6];
}

function getBrainTopology() {
  const inputs = 11;
  const outputs = 2;
  const hidden = parseHiddenLayers();
  return [inputs, ...hidden, outputs];
}

function handleGenerationComplete() {
  const popSize = population.length;
  let bestFitness = 0;
  let totalFitness = 0;
  
  for (let bee of population) {
    if (bee.fitness > bestFitness) bestFitness = bee.fitness;
    totalFitness += bee.fitness;
  }
  
  const avgFitness = totalFitness / popSize;
  const successRate = (sim.successCount / popSize) * 100;
  
  history.push({
    generation: generation,
    bestFitness: bestFitness,
    avgFitness: avgFitness
  });
  
  drawProgressChart(canvasChart, history);
  
  const mutationRate = parseInt(inputMutationRate.value, 10) / 100;
  const mutationAmount = parseFloat(inputMutationAmount.value);
  const elitePct = parseInt(inputElitePct.value, 10);
  const selectionMethod = selectSelection.value;
  const desiredPopSize = parseInt(inputPopSize.value, 10);
  const topology = getBrainTopology();
  
  population = nextGeneration(
    population,
    desiredPopSize,
    mutationRate,
    mutationAmount,
    elitePct,
    selectionMethod,
    sim.startX,
    sim.startY,
    topology
  );
  
  generation++;
  sim.generation = generation;
  sim.startGeneration(population);
  
  if (isSimulationRunning) {
    sim.isPaused = false;
  }
  
  updateUIStats(bestFitness, avgFitness, successRate);
}

function updateUIStats(lastBest = null, lastAvg = null, lastSuccess = null) {
  textGeneration.innerText = generation;
  textHeaderGen.innerText = generation;
  
  if (lastBest !== null) {
    textBest.innerText = lastBest.toFixed(1);
    textHeaderBest.innerText = lastBest.toFixed(1);
  }
  if (lastAvg !== null) {
    textAvg.innerText = lastAvg.toFixed(1);
  }
  if (lastSuccess !== null) {
    textSuccess.innerText = `${lastSuccess.toFixed(0)}%`;
    textHeaderSuccess.innerText = `${lastSuccess.toFixed(0)}%`;
  }
}

// Setup simulation instance
sim = new Simulation(
  canvasSim,
  handleGenerationComplete,
  function onStep() {
    const aliveBees = sim.activeCount;
    const popSize = population.length;
    
    // Choose appropriate label based on active theme
    let word = 'Bees';
    if (currentTheme === 'ants') word = 'Ants';
    else if (currentTheme === 'drones') word = 'Drones';
    
    textActiveCount.innerText = `${word} Alive: ${aliveBees}/${popSize}`;
    
    const ratio = Math.min(100, (sim.currentStep / sim.maxGenSteps) * 100);
    progressBarGenTime.style.width = `${ratio}%`;
    
    const secElapsed = (sim.currentStep / 60).toFixed(1);
    const secMax = (sim.maxGenSteps / 60).toFixed(1);
    textTime.innerText = `${secElapsed}s / ${secMax}s`;
    
    if (sim.bestBee) {
      const bestIdx = population.indexOf(sim.bestBee) + 1;
      let label = 'Bee';
      if (currentTheme === 'ants') label = 'Ant';
      else if (currentTheme === 'drones') label = 'Drone';
      
      visualizedBeeId.innerText = `${label} #${bestIdx} (${sim.bestBee.hasSucceeded ? 'Success' : 'Active'})`;
    }
  }
);

window.addEventListener('resize', () => {
  sim.init();
});

sim.loadPresetMap('simple-maze');

function initializePopulation() {
  generation = 1;
  sim.generation = generation;
  history = [];
  
  const popSize = parseInt(inputPopSize.value, 10);
  const topology = getBrainTopology();
  
  population = [];
  for (let i = 0; i < popSize; i++) {
    population.push(new Bee(sim.startX, sim.startY, topology));
  }
  
  sim.startGeneration(population);
  updateUIStats(0, 0, 0);
  drawProgressChart(canvasChart, history);
}

// --- Dynamic Theme UI Syncer ---
function updateThemeUI() {
  const h1 = document.querySelector('.logo h1');
  const icon = document.querySelector('.logo .icon');
  const panelHeader = document.querySelector('.nn-panel .panel-header h3');
  const activeLabel = document.getElementById('txt-active-count');
  
  const popSize = population.length;
  
  if (currentTheme === 'bees') {
    document.title = 'CyberBee - Neuroevolution Drone Simulator';
    h1.innerHTML = 'CYBER<span class="highlight">BEE</span>';
    icon.innerText = '🐝';
    panelHeader.innerHTML = '🧠 Best Bee\'s Neural Activation';
    activeLabel.innerText = `Bees Alive: ${sim.activeCount}/${popSize}`;
  } else if (currentTheme === 'ants') {
    document.title = 'CyberAnt - Neuroevolution Agent Simulator';
    h1.innerHTML = 'CYBER<span class="highlight">ANT</span>';
    icon.innerText = '🐜';
    panelHeader.innerHTML = '🧠 Best Ant\'s Neural Activation';
    activeLabel.innerText = `Ants Alive: ${sim.activeCount}/${popSize}`;
  } else if (currentTheme === 'drones') {
    document.title = 'CyberDrone - Neuroevolution Drone Simulator';
    h1.innerHTML = 'CYBER<span class="highlight">DRONE</span>';
    icon.innerText = '🛸';
    panelHeader.innerHTML = '🧠 Best Drone\'s Neural Activation';
    activeLabel.innerText = `Drones Alive: ${sim.activeCount}/${popSize}`;
  }
}

// Theme Dropdown listener
selectTheme.addEventListener('change', () => {
  currentTheme = selectTheme.value;
  sim.theme = currentTheme;
  updateThemeUI();
  sim.draw(); // Force redraw scene
});

// Sync initial theme
currentTheme = selectTheme.value;
sim.theme = currentTheme;
updateThemeUI();

initializePopulation();

canvasSim.addEventListener('mousedown', (e) => {
  const pos = getMousePos(canvasSim, e);
  if (e.ctrlKey) {
    deleteObstacleAt(pos);
    return;
  }
  
  isDrawing = true;
  drawStart = pos;
  drawCurrent = pos;
  lastBrushPos = pos;
  
  if (activeTool === 'target') {
    sim.target.x = pos.x;
    sim.target.y = pos.y;
    isDrawing = false;
  } else if (activeTool === 'eraser') {
    deleteObstacleAt(pos);
  } else if (activeTool === 'brush') {
    addBrushWallAt(pos);
  }
});

canvasSim.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  const pos = getMousePos(canvasSim, e);
  drawCurrent = pos;
  
  if (activeTool === 'brush') {
    addBrushWallAt(pos);
  } else if (activeTool === 'eraser') {
    deleteObstacleAt(pos);
  }
});

canvasSim.addEventListener('mouseup', (e) => {
  if (!isDrawing) return;
  isDrawing = false;
  
  const pos = getMousePos(canvasSim, e);
  if (activeTool === 'rect') {
    const x = Math.min(drawStart.x, pos.x);
    const y = Math.min(drawStart.y, pos.y);
    const w = Math.abs(drawStart.x - pos.x);
    const h = Math.abs(drawStart.y - pos.y);
    if (w > 5 && h > 5) {
      sim.obstacles.push(new RectObstacle(x, y, w, h));
    }
  } else if (activeTool === 'circle') {
    const r = Math.hypot(drawStart.x - pos.x, drawStart.y - pos.y);
    if (r > 4) {
      sim.obstacles.push(new CircleObstacle(drawStart.x, drawStart.y, r));
    }
  }
});

function deleteObstacleAt(pos) {
  for (let i = sim.obstacles.length - 1; i >= 0; i--) {
    if (sim.obstacles[i].containsPoint(pos.x, pos.y)) {
      sim.obstacles.splice(i, 1);
      break;
    }
  }
}

function addBrushWallAt(pos) {
  const brushSize = parseInt(sliderBrushSize.value, 10);
  if (lastBrushPos) {
    const dist = Math.hypot(pos.x - lastBrushPos.x, pos.y - lastBrushPos.y);
    if (dist < brushSize * 0.6) return;
  }
  sim.obstacles.push(new CircleObstacle(pos.x, pos.y, brushSize / 2));
  lastBrushPos = pos;
}

chkShowSensors.addEventListener('change', () => sim.showSensors = chkShowSensors.checked);
chkShowBest.addEventListener('change', () => sim.showBest = chkShowBest.checked);
chkShowTrails.addEventListener('change', () => sim.showTrails = chkShowTrails.checked);

toolButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    toolButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeTool = btn.dataset.tool;
  });
});

btnPlayPause.addEventListener('click', () => {
  isSimulationRunning = !isSimulationRunning;
  sim.isPaused = !isSimulationRunning;
  
  if (isSimulationRunning) {
    txtPlayPause.innerText = 'Pause';
    btnPlayPause.classList.remove('btn-primary');
    btnPlayPause.classList.add('btn-secondary');
    overlay.classList.remove('active');
  } else {
    txtPlayPause.innerText = 'Resume';
    btnPlayPause.classList.remove('btn-secondary');
    btnPlayPause.classList.add('btn-primary');
    overlay.classList.add('active');
  }
});

btnNextGen.addEventListener('click', () => {
  sim.currentStep = sim.maxGenSteps;
  sim.isPaused = false;
  if (!isSimulationRunning) {
    sim.update();
  }
});

btnReset.addEventListener('click', () => {
  initializePopulation();
  isSimulationRunning = false;
  sim.isPaused = true;
  txtPlayPause.innerText = 'Start';
  btnPlayPause.classList.remove('btn-secondary');
  btnPlayPause.classList.add('btn-primary');
  overlay.classList.add('active');
});

btnClearObstacles.addEventListener('click', () => {
  sim.obstacles = [];
});

sliderSpeed.addEventListener('input', () => {
  const val = parseInt(sliderSpeed.value, 10);
  sim.speed = val;
  valSpeed.innerText = `${val}x`;
});

selectPreset.addEventListener('change', () => {
  sim.loadPresetMap(selectPreset.value);
  sim.startGeneration(population);
});

sliderBrushSize.addEventListener('input', () => {
  valBrushSize.innerText = `${sliderBrushSize.value}px`;
});

inputPopSize.addEventListener('input', () => {
  valPopSize.innerText = inputPopSize.value;
});

inputMutationRate.addEventListener('input', () => {
  valMutationRate.innerText = `${inputMutationRate.value}%`;
});

inputMutationAmount.addEventListener('input', () => {
  valMutationAmount.innerText = inputMutationAmount.value;
});

inputElitePct.addEventListener('input', () => {
  valElitePct.innerText = `${inputElitePct.value}%`;
});

function loop() {
  sim.update();
  sim.draw();
  
  if (isDrawing || activeTool === 'brush' || activeTool === 'eraser' || activeTool === 'rect' || activeTool === 'circle') {
    drawMouseOverlay();
  }
  
  if (sim.bestBee && sim.bestBee.lastActivations) {
    drawNeuralNetwork(canvasNN, sim.bestBee.lastActivations, sim.bestBee.brain.layers);
  } else {
    drawNeuralNetwork(canvasNN, null, null);
  }
  
  requestAnimationFrame(loop);
}

function drawMouseOverlay() {
  const ctx = sim.ctx;
  ctx.save();
  
  if (isDrawing && activeTool === 'rect') {
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.rect(drawStart.x, drawStart.y, drawCurrent.x - drawStart.x, drawCurrent.y - drawStart.y);
    ctx.stroke();
  } else if (isDrawing && activeTool === 'circle') {
    const r = Math.hypot(drawStart.x - drawCurrent.x, drawStart.y - drawCurrent.y);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(drawStart.x, drawStart.y, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

let canvasHoverPos = { x: -100, y: -100 };
canvasSim.addEventListener('mousemove', (e) => {
  canvasHoverPos = getMousePos(canvasSim, e);
});
canvasSim.addEventListener('mouseleave', () => {
  canvasHoverPos = { x: -100, y: -100 };
});

function drawCursorIndicator() {
  if (canvasHoverPos.x < 0 || canvasHoverPos.y < 0) return;
  
  const ctx = sim.ctx;
  ctx.save();
  
  if (activeTool === 'brush') {
    const brushSize = parseInt(sliderBrushSize.value, 10);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(canvasHoverPos.x, canvasHoverPos.y, brushSize / 2, 0, Math.PI * 2);
    ctx.stroke();
  } else if (activeTool === 'eraser') {
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
    ctx.fillStyle = 'rgba(244, 63, 94, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(canvasHoverPos.x, canvasHoverPos.y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

const originalDraw = sim.draw;
sim.draw = function() {
  originalDraw.apply(this);
  drawCursorIndicator();
};

requestAnimationFrame(loop);
overlay.classList.add('active');
updateUIStats(0, 0, 0);
updateThemeUI();
