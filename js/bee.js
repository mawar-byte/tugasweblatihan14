import { NeuralNetwork } from './nn.js';

export class Bee {
  /**
   * @param {number} startX
   * @param {number} startY
   * @param {number[]} brainTopology
   * @param {NeuralNetwork} [existingBrain]
   */
  constructor(startX, startY, brainTopology, existingBrain = null) {
    this.startX = startX;
    this.startY = startY;
    this.x = startX;
    this.y = startY;
    this.vx = 0;
    this.vy = 0;
    this.angle = -Math.PI / 2; // Pointing upwards initially
    this.radius = 8;
    
    this.maxSpeed = 4;
    this.maxForce = 0.25;
    this.maxSteer = 0.12; // Radians per frame
    
    // Brain (Neural Network)
    this.brainTopology = brainTopology;
    this.brain = existingBrain ? existingBrain.copy() : new NeuralNetwork(brainTopology);
    this.lastActivations = null; // Stored from feedForward for visualizer
    
    // State
    this.isDead = false;
    this.hasSucceeded = false;
    this.stepsTaken = 0;
    this.closestDist = Infinity;
    this.startDist = 0; // Set upon simulation start
    this.fitness = 0;
    
    // Raycasting Sensors: 7 rays at angles (in radians) relative to heading
    this.sensorAngles = [
      -Math.PI / 2,     // -90 degrees (left)
      -Math.PI / 4,     // -45 degrees (forward-left)
      -Math.PI / 12,    // -15 degrees (forward-left-narrow)
      0,                // 0 degrees (front)
      Math.PI / 12,     // 15 degrees (forward-right-narrow)
      Math.PI / 4,      // 45 degrees (forward-right)
      Math.PI / 2       // 90 degrees (right)
    ];
    this.sensorLength = 160;
    this.sensorReadings = new Array(this.sensorAngles.length).fill(1.0);
    this.sensorPoints = new Array(this.sensorAngles.length).fill(null);
    
    // Visuals
    this.trail = [];
    this.maxTrailLength = 20;
    this.wingPhase = Math.random() * Math.PI * 2;
  }

  /**
   * Update the bee physics, sensors, brain and collision states.
   */
  update(obstacles, target, boundaryWidth, boundaryHeight, currentStep) {
    if (this.isDead || this.hasSucceeded) return;
    
    this.stepsTaken = currentStep;
    
    // 1. Update Distance Metrics
    const distToTarget = Math.hypot(target.x - this.x, target.y - this.y);
    if (this.startDist === 0) {
      this.startDist = distToTarget;
    }
    if (distToTarget < this.closestDist) {
      this.closestDist = distToTarget;
    }
    
    // 2. Check success (reached target)
    if (distToTarget < this.radius + target.radius) {
      this.hasSucceeded = true;
      this.calculateFitness();
      return;
    }
    
    // 3. Update Sensors (Raycasting)
    this.updateSensors(obstacles, boundaryWidth, boundaryHeight);
    
    // 4. Gather Brain Inputs
    // Total inputs = 11:
    //   - 7 normalized sensor readings (0 = wall, 1 = clear)
    //   - 2 unit vector components to target (dx/dist, dy/dist)
    //   - 2 normalized velocity components (vx/maxSpeed, vy/maxSpeed)
    const inputs = [];
    // Sensors (7)
    for (let reading of this.sensorReadings) {
      inputs.push(reading);
    }
    // Target Unit Vector (2)
    inputs.push((target.x - this.x) / distToTarget);
    inputs.push((target.y - this.y) / distToTarget);
    // Velocity (2)
    inputs.push(this.vx / this.maxSpeed);
    inputs.push(this.vy / this.maxSpeed);
    
    // 5. Query Brain
    this.lastActivations = this.brain.feedForward(inputs);
    const outputs = this.lastActivations[this.lastActivations.length - 1];
    
    // 6. Apply Steering & Forces (Outputs)
    const steerForce = outputs[0] * this.maxSteer; // Steering rotation
    const thrust = outputs[1]; // Thrust speed control
    
    // Update orientation
    this.angle += steerForce;
    
    // Apply thrust force along heading
    let accelX = Math.cos(this.angle) * thrust * this.maxForce;
    let accelY = Math.sin(this.angle) * thrust * this.maxForce;
    
    // Update velocity with acceleration
    this.vx += accelX;
    this.vy += accelY;
    
    // Damping (Friction) mimicking flight drag
    this.vx *= 0.97;
    this.vy *= 0.97;
    
    // Speed clamp
    const speed = Math.hypot(this.vx, this.vy);
    if (speed > this.maxSpeed) {
      this.vx = (this.vx / speed) * this.maxSpeed;
      this.vy = (this.vy / speed) * this.maxSpeed;
    }
    
    // Update position
    this.x += this.vx;
    this.y += this.vy;
    
    // Record flight trail
    this.trail.push({ x: this.x, y: this.y });
    if (this.trail.length > this.maxTrailLength) {
      this.trail.shift();
    }
    
    // Flutter wings animation
    this.wingPhase += 0.4 + Math.abs(thrust) * 0.4;
    
    // 7. Collision Checks
    this.checkCollisions(obstacles, boundaryWidth, boundaryHeight);
    
    // Timeout check (if it takes too long or remains stationary, penalize)
    if (this.stepsTaken > 900) {
      this.isDead = true;
      this.calculateFitness();
    }
  }

  /**
   * Run raycasting for sensors.
   */
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
      
      // Check boundary intersection
      const bounds = [
        // Top boundary
        { p3: { x: 0, y: 0 }, p4: { x: boundaryWidth, y: 0 } },
        // Bottom boundary
        { p3: { x: 0, y: boundaryHeight }, p4: { x: boundaryWidth, y: boundaryHeight } },
        // Left boundary
        { p3: { x: 0, y: 0 }, p4: { x: 0, y: boundaryHeight } },
        // Right boundary
        { p3: { x: boundaryWidth, y: 0 }, p4: { x: boundaryWidth, y: boundaryHeight } }
      ];
      
      for (let bound of bounds) {
        const d = intersectRayToLine(startPoint, endPoint, bound.p3, bound.p4);
        if (d && d.t < minT) {
          minT = d.t;
          closestPoint = d;
        }
      }
      
      // Check obstacle intersection
      const minX = Math.min(startPoint.x, endPoint.x);
      const maxX = Math.max(startPoint.x, endPoint.x);
      const minY = Math.min(startPoint.y, endPoint.y);
      const maxY = Math.max(startPoint.y, endPoint.y);
      
      for (let obs of obstacles) {
        // AABB bounding box check to optimize performance
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
      
      this.sensorReadings[i] = minT; // 0 (immediate) to 1 (clear)
      this.sensorPoints[i] = closestPoint;
    }
  }

  /**
   * Checks collisions with obstacles and boundaries.
   */
  checkCollisions(obstacles, boundaryWidth, boundaryHeight) {
    // 1. Boundary check
    if (this.x - this.radius < 0 || this.x + this.radius > boundaryWidth ||
        this.y - this.radius < 0 || this.y + this.radius > boundaryHeight) {
      this.isDead = true;
      this.calculateFitness();
      return;
    }
    
    // 2. Obstacles check
    for (let obs of obstacles) {
      if (obs.collidesWithCircle(this.x, this.y, this.radius)) {
        this.isDead = true;
        this.calculateFitness();
        return;
      }
    }
  }

  /**
   * Evaluate fitness for genetics.
   */
  calculateFitness() {
    if (this.hasSucceeded) {
      // Reward success and speed!
      // Faster time = higher fitness
      // Maximum duration is around 900 steps
      this.fitness = 3000 + Math.max(0, 1000 - this.stepsTaken);
    } else {
      // Progress based fitness
      // We reward getting closer to the target than they started
      const progress = 1.0 - (this.closestDist / this.startDist);
      this.fitness = Math.max(0.1, progress * 1000);
      
      // Penalty for dying via collision
      this.fitness *= 0.65;
    }
  }

  /**
   * Draw the bee drone and optionally its sensors & highlight.
   */
  draw(ctx, showSensors, isBest) {
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
        
        // Draw collision point dot
        if (this.sensorReadings[i] < 1.0) {
          ctx.fillStyle = isClose ? '#ef4444' : '#06b6d4';
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // 2. Draw flight trail
    if (this.trail.length > 1) {
      ctx.strokeStyle = isBest ? 'rgba(245, 158, 11, 0.25)' : 'rgba(99, 102, 241, 0.1)';
      ctx.lineWidth = isBest ? 2 : 1.5;
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
      ctx.shadowColor = 'rgba(245, 158, 11, 0.8)';
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius + 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.shadowBlur = 0; // Reset shadow
    }
    
    // If dead, draw explosion cross/dust
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
    
    // If succeeded, draw success halo
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
    
    // 4. Draw Bee Body (Cyber-styling)
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);
    
    // Fluttering Wings
    ctx.fillStyle = 'rgba(200, 230, 255, 0.65)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 1;
    
    const wingSizeX = 11;
    const wingSizeY = 5;
    const wingSpread = Math.sin(this.wingPhase) * 0.4;
    
    // Left Wing
    ctx.save();
    ctx.translate(0, -2);
    ctx.rotate(-Math.PI / 3 + wingSpread);
    ctx.beginPath();
    ctx.ellipse(0, -wingSizeX/2, wingSizeY, wingSizeX, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    
    // Right Wing
    ctx.save();
    ctx.translate(0, 2);
    ctx.rotate(Math.PI / 3 - wingSpread);
    ctx.beginPath();
    ctx.ellipse(0, wingSizeX/2, wingSizeY, wingSizeX, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    
    // Bee Body (Stripe pattern)
    ctx.fillStyle = '#1e293b'; // Cyber dark armor base
    ctx.beginPath();
    ctx.ellipse(0, 0, 11, 7, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Cyber yellow stripes
    ctx.fillStyle = '#f59e0b';
    ctx.fillRect(-4, -6.5, 3, 13);
    ctx.fillRect(2, -6, 3, 12);
    
    // Eyes/Sensor head (glowing front)
    ctx.fillStyle = '#06b6d4'; // Cyan sensor visor
    ctx.beginPath();
    ctx.arc(8, 0, 4, -Math.PI / 2, Math.PI / 2);
    ctx.fill();
    
    ctx.restore();
  }
}

/**
 * Line to segment intersection helper.
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
