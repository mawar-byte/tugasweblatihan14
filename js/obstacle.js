/**
 * Segment-to-segment intersection helper.
 * Ray start: p1, Ray end: p2
 * Segment start: p3, Segment end: p4
 */
export function intersectSegments(p1, p2, p3, p4) {
  const denominator = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
  if (denominator === 0) return null; // Parallel

  const ua = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)) / denominator;
  const ub = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) / denominator;

  // Intersects if intersection point lies within both segments
  if (ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1) {
    return {
      x: p1.x + ua * (p2.x - p1.x),
      y: p1.y + ua * (p2.y - p1.y),
      t: ua // distance ratio along the ray (0 to 1)
    };
  }
  return null;
}

/**
 * Base Obstacle class with motion support.
 */
export class Obstacle {
  constructor(type) {
    this.type = type;
    this.vx = 0;
    this.vy = 0;
    this.minX = null;
    this.maxX = null;
    this.minY = null;
    this.maxY = null;
  }
  
  // Set motion constraints
  setMotion(vx, vy, minX, maxX, minY, maxY) {
    this.vx = vx;
    this.vy = vy;
    this.minX = minX;
    this.maxX = maxX;
    this.minY = minY;
    this.maxY = maxY;
  }

  // Update moving obstacles
  update() {
    if (this.vx === 0 && this.vy === 0) return;
    
    this.x += this.vx;
    this.y += this.vy;
    
    // Bounce X
    if (this.minX !== null && this.x < this.minX) {
      this.x = this.minX;
      this.vx *= -1;
    } else if (this.maxX !== null && this.x > this.maxX) {
      this.x = this.maxX;
      this.vx *= -1;
    }
    
    // Bounce Y
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

/**
 * Rectangular Obstacle
 */
export class RectObstacle extends Obstacle {
  constructor(x, y, width, height) {
    super('rect');
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  }

  draw(ctx) {
    ctx.save();
    
    // Glassmorphic / Cyberpunk Obstacle styling
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.5)';
    ctx.fillStyle = 'rgba(30, 9, 15, 0.7)';
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    
    // Draw box
    ctx.beginPath();
    ctx.rect(this.x, this.y, this.width, this.height);
    ctx.fill();
    ctx.stroke();
    
    // Dynamic interior stripes (tech look)
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.15)';
    ctx.lineWidth = 1;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    for (let offset = 5; offset < this.width + this.height; offset += 15) {
      // Draw diagonal lines
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

    // Rectangle's 4 boundary lines
    const edges = [
      [r1, r2], // Top
      [r2, r3], // Right
      [r3, r4], // Bottom
      [r4, r1]  // Left
    ];

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

/**
 * Circular Obstacle
 */
export class CircleObstacle extends Obstacle {
  constructor(x, y, radius) {
    super('circle');
    this.x = x;
    this.y = y;
    this.radius = radius;
  }

  draw(ctx) {
    ctx.save();
    
    // Styling
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.5)';
    ctx.fillStyle = 'rgba(30, 9, 15, 0.7)';
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    
    // Draw crosshair pattern inside circle for sci-fi look
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
    if (discriminant < 0) {
      return null; // No intersection
    }

    const t1 = (-b - Math.sqrt(discriminant)) / (2 * a);
    const t2 = (-b + Math.sqrt(discriminant)) / (2 * a);

    // We want the smallest t in [0, 1]
    if (t1 >= 0 && t1 <= 1) {
      return {
        x: p1.x + t1 * dX,
        y: p1.y + t1 * dY,
        t: t1
      };
    }
    if (t2 >= 0 && t2 <= 1) {
      return {
        x: p1.x + t2 * dX,
        y: p1.y + t2 * dY,
        t: t2
      };
    }

    return null;
  }
}
