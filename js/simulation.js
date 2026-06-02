import { Bee } from './bee.js';
import { RectObstacle, CircleObstacle } from './obstacle.js';

export class Simulation {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {Function} onGenCompleteCallback - Fired when generation ends.
   * @param {Function} onStepCallback - Fired every simulation tick.
   */
  constructor(canvas, onGenCompleteCallback, onStepCallback) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    
    this.bees = [];
    this.obstacles = [];
    this.target = { x: 0, y: 0, radius: 12 };
    
    this.startX = 0;
    this.startY = 0;
    
    this.currentStep = 0;
    this.maxGenSteps = 900; // 15 seconds at 60 FPS
    this.generation = 1;
    
    this.isPaused = true;
    this.speed = 1; // Simulation speed scaling (1x to 10x)
    
    // Toggles
    this.showSensors = true;
    this.showBest = true;
    this.showTrails = true;
    
    // Callbacks
    this.onGenerationComplete = onGenCompleteCallback;
    this.onStep = onStepCallback;
    
    // Stats
    this.activeCount = 0;
    this.bestBee = null;
    this.successCount = 0;
    
    this.init();
  }

  /**
   * Initialize or handle resizing of the simulation canvas.
   */
  init() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
    
    // Standard starting positions
    this.startX = this.canvas.width / 2;
    this.startY = this.canvas.height - 50;
    
    // Standard target position (top middle)
    this.target.x = this.canvas.width / 2;
    this.target.y = 70;
  }

  /**
   * Start a new simulation run with a given population.
   * @param {Bee[]} population
   */
  startGeneration(population) {
    this.bees = population;
    this.currentStep = 0;
    this.activeCount = population.length;
    this.successCount = 0;
    this.bestBee = population[0];
    
    // Set initial target distance on all bees
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

  /**
   * Update the simulation physics loop.
   */
  update() {
    if (this.isPaused || this.bees.length === 0) return;

    // Run physics updates multiple times per frame for speed multiplier
    for (let s = 0; s < this.speed; s++) {
      this.currentStep++;
      this.activeCount = 0;
      this.successCount = 0;
      
      // Update moving obstacles
      for (let obs of this.obstacles) {
        obs.update();
      }
      
      let currentBest = null;

      for (let bee of this.bees) {
        // Update physics and brain
        bee.update(this.obstacles, this.target, this.canvas.width, this.canvas.height, this.currentStep);
        
        if (!bee.isDead && !bee.hasSucceeded) {
          this.activeCount++;
        }
        if (bee.hasSucceeded) {
          this.successCount++;
        }
        
        // Determine best performing bee currently
        // Standardize fitness value during evaluation
        bee.calculateFitness();
        if (!currentBest || bee.fitness > currentBest.fitness) {
          currentBest = bee;
        }
      }
      
      if (currentBest) {
        this.bestBee = currentBest;
      }
      
      // Notify step callback
      if (this.onStep) {
        this.onStep();
      }

      // Check if generation is finished
      if (this.activeCount === 0 || this.currentStep >= this.maxGenSteps) {
        // Finalize fitnesses
        for (let bee of this.bees) {
          bee.calculateFitness();
        }
        this.isPaused = true;
        
        if (this.onGenerationComplete) {
          this.onGenerationComplete();
        }
        break; // Stop updating if gen ended
      }
    }
  }

  /**
   * Renders the simulation scene.
   */
  draw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // 1. Draw Target (Glowing cybernetic orb)
    this.ctx.save();
    const pulse = 1.0 + Math.sin(Date.now() / 150) * 0.08;
    
    // Glow effect
    this.ctx.shadowBlur = 20 * pulse;
    this.ctx.shadowColor = 'rgba(16, 185, 129, 0.8)';
    this.ctx.fillStyle = '#10b981';
    
    this.ctx.beginPath();
    this.ctx.arc(this.target.x, this.target.y, this.target.radius * pulse, 0, Math.PI * 2);
    this.ctx.fill();
    
    // Rotating outer tech ring
    this.ctx.shadowBlur = 0;
    this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    this.ctx.lineWidth = 1.5;
    this.ctx.beginPath();
    this.ctx.arc(this.target.x, this.target.y, this.target.radius + 8, 0, Math.PI * 2);
    this.ctx.stroke();
    
    // Pulsing dashed ring
    this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)';
    this.ctx.setLineDash([4, 4]);
    this.ctx.save();
    this.ctx.translate(this.target.x, this.target.y);
    this.ctx.rotate(Date.now() / 800);
    this.ctx.beginPath();
    this.ctx.arc(0, 0, this.target.radius + 13, 0, Math.PI * 2);
    this.ctx.stroke();
    this.ctx.restore();
    
    ctx.restore();

    // 2. Draw Start Position (Hangar landing pad)
    this.ctx.save();
    this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.35)';
    this.ctx.lineWidth = 1;
    this.ctx.setLineDash([3, 5]);
    this.ctx.beginPath();
    this.ctx.arc(this.startX, this.startY, 16, 0, Math.PI * 2);
    this.ctx.stroke();
    
    // Hangar crosshairs
    this.ctx.setLineDash([]);
    this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
    this.ctx.beginPath();
    this.ctx.moveTo(this.startX - 22, this.startY);
    this.ctx.lineTo(this.startX + 22, this.startY);
    this.ctx.moveTo(this.startX, this.startY - 22);
    this.ctx.lineTo(this.startX, this.startY + 22);
    this.ctx.stroke();
    
    // Draw "H" label
    this.ctx.fillStyle = 'rgba(99, 102, 241, 0.4)';
    this.ctx.font = '700 10px var(--font-sans)';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText('H', this.startX, this.startY);
    this.ctx.restore();

    // 3. Draw Obstacles
    for (let obs of this.obstacles) {
      obs.draw(this.ctx);
    }

    // 4. Draw Bees (Sorted to draw dead first, then succeeded, then alive, then best on top)
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
      bee.draw(this.ctx, this.showSensors, isBest);
    }
  }

  /**
   * Loads a preset map configuration onto the canvas.
   * @param {string} name
   */
  loadPresetMap(name) {
    this.obstacles = [];
    const w = this.canvas.width;
    const h = this.canvas.height;
    
    // Reset target/start standard positions
    this.target.x = w / 2;
    this.target.y = 70;
    this.startX = w / 2;
    this.startY = h - 50;

    switch (name) {
      case 'simple-maze': {
        // Vertical blocks on sides and a horizontal gate in middle
        this.obstacles.push(new RectObstacle(w * 0.15, h * 0.45, w * 0.3, 30)); // Left block
        this.obstacles.push(new RectObstacle(w * 0.55, h * 0.45, w * 0.3, 30)); // Right block
        
        // Dynamic barrier block above start
        this.obstacles.push(new CircleObstacle(w / 2, h * 0.72, 35));
        break;
      }
      case 'zigzag': {
        // 3 horizontal baffles that alternate sides
        this.obstacles.push(new RectObstacle(0, h * 0.3, w * 0.7, 25));       // Top-left
        this.obstacles.push(new RectObstacle(w * 0.3, h * 0.52, w * 0.7, 25)); // Middle-right
        this.obstacles.push(new RectObstacle(0, h * 0.72, w * 0.6, 25));       // Bottom-left
        break;
      }
      case 'dynamic-gate': {
        // Dynamic moving rectangle gates sliding side-to-side
        const gate1 = new RectObstacle(w * 0.3, h * 0.38, w * 0.4, 25);
        gate1.setMotion(2.5, 0, w * 0.05, w * 0.55, null, null); // Slides horizontally
        
        const gate2 = new RectObstacle(w * 0.1, h * 0.62, w * 0.35, 25);
        gate2.setMotion(-3, 0, w * 0.05, w * 0.6, null, null); // Slides horizontally opposite
        
        this.obstacles.push(gate1);
        this.obstacles.push(gate2);
        
        // Static circle obstacle in the center
        this.obstacles.push(new CircleObstacle(w / 2, h * 0.5, 20));
        break;
      }
      case 'ring-fire': {
        // A ring of circles in the center of the arena
        const cx = w / 2;
        const cy = h / 2 - 20;
        const count = 6;
        const dist = 90;
        
        for (let i = 0; i < count; i++) {
          // Leave a dynamic gap by moving one of the circles
          const angle = (i * Math.PI * 2) / count;
          const ox = cx + Math.cos(angle) * dist;
          const oy = cy + Math.sin(angle) * dist;
          const circle = new CircleObstacle(ox, oy, 22);
          
          if (i === 0 || i === 3) {
            // Animate these vertical bobbing circles
            circle.setMotion(0, 1.2, null, null, oy - 25, oy + 25);
          }
          this.obstacles.push(circle);
        }
        
        // Add a central circular obstacle
        this.obstacles.push(new CircleObstacle(cx, cy, 18));
        break;
      }
      case 'empty':
      default:
        // No obstacles
        break;
    }
  }
}
