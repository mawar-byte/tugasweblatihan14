/**
 * Visualizes a Neural Network's activations, weights, and structure on a Canvas.
 * 
 * @param {HTMLCanvasElement} canvas
 * @param {number[][]} activations - Activations for each layer (from feedForward).
 * @param {Layer[]} layers - The weights and biases layers from the network.
 */
export function drawNeuralNetwork(canvas, activations, layers) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // Handle DPI scaling
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
  
  // Calculate coordinates of all nodes
  const paddingX = 45;
  const paddingY = 20;
  const layerSpacingX = (width - paddingX * 2) / (layerCount - 1);
  
  const nodes = []; // 2D array of {x, y} coordinates for each node
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

  // 1. Draw Connections (Weights)
  for (let l = 0; l < layers.length; l++) {
    const currentLayer = layers[l];
    const prevNodes = nodes[l];
    const nextNodes = nodes[l + 1];
    
    for (let i = 0; i < currentLayer.outputSize; i++) {
      const nextNode = nextNodes[i];
      for (let j = 0; j < currentLayer.inputSize; j++) {
        const prevNode = prevNodes[j];
        const weight = currentLayer.weights[i][j];
        
        // Color: positive is cyber cyan/blue, negative is neon red/pink
        ctx.lineWidth = Math.min(Math.abs(weight) * 1.5, 3);
        if (ctx.lineWidth < 0.1) continue; // Skip near-zero connections
        
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

  // 2. Draw Moving Signals (Micro-animations representing feedforward data flow)
  const animTime = (Date.now() / 1500) % 1.0; // Loops every 1.5s
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
        
        // Only flow signal if there's substantial activation and weight
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

  // 3. Draw Nodes (Neurons)
  const inputLabels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'T_dx', 'T_dy', 'V_vx', 'V_vy'];
  const outputLabels = ['Steer', 'Thrust'];
  
  for (let l = 0; l < layerCount; l++) {
    const size = layerSizes[l];
    for (let i = 0; i < size; i++) {
      const node = nodes[l][i];
      const activation = activations[l][i];
      
      // Node glow & fill based on activation value
      ctx.save();
      
      const nodeRadius = 5.5;
      ctx.shadowBlur = Math.abs(activation) * 8;
      
      // Activation colors: Amber yellow for positive, Cyan for negative, dark gray for zero
      if (activation > 0.05) {
        ctx.fillStyle = `rgba(245, 158, 11, ${0.4 + activation * 0.6})`; // Amber
        ctx.shadowColor = '#f59e0b';
      } else if (activation < -0.05) {
        ctx.fillStyle = `rgba(6, 182, 212, ${0.4 + Math.abs(activation) * 0.6})`; // Cyan
        ctx.shadowColor = '#06b6d4';
      } else {
        ctx.fillStyle = '#334155'; // Neutral grey
        ctx.shadowBlur = 0;
      }
      
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      
      ctx.beginPath();
      ctx.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      
      // Draw labels for input and output layer
      if (l === 0) {
        // Input labels (to the left)
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.font = '8px var(--font-mono)';
        ctx.textAlign = 'right';
        ctx.fillText(inputLabels[i] || `In${i}`, node.x - 8, node.y + 3);
      } else if (l === layerCount - 1) {
        // Output labels (to the right)
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.font = '9px var(--font-sans)';
        ctx.textAlign = 'left';
        ctx.fillText(outputLabels[i] || `Out${i}`, node.x + 8, node.y + 3);
      }
    }
  }
}

/**
 * Custom line chart to display best & average fitness progression.
 * 
 * @param {HTMLCanvasElement} canvas
 * @param {Array<{generation: number, bestFitness: number, avgFitness: number}>} history
 */
export function drawProgressChart(canvas, history) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // Handle DPI scaling
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

  // 1. Determine Y Scale limits (Min 0, Max maxFitness or at least 100)
  let maxFitness = 100;
  for (let record of history) {
    if (record.bestFitness > maxFitness) maxFitness = record.bestFitness;
    if (record.avgFitness > maxFitness) maxFitness = record.avgFitness;
  }
  maxFitness *= 1.1; // Add 10% headroom

  // Determine X Scale (Generations)
  const maxGen = Math.max(1, history[history.length - 1].generation);
  const minGen = 1;
  const genSpan = maxGen - minGen;

  // Helper to map values to canvas coordinates
  const getX = (gen) => {
    if (genSpan === 0) return paddingLeft + chartWidth / 2;
    return paddingLeft + ((gen - minGen) / genSpan) * chartWidth;
  };
  const getY = (fitness) => {
    return paddingTop + chartHeight - (fitness / maxFitness) * chartHeight;
  };

  // 2. Draw Grid Lines and Labels
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  
  // Horizontal grid lines (4 divisions)
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
    
    // Draw Y label
    ctx.fillText(val.toFixed(0), paddingLeft - 6, y);
  }

  // Vertical grid lines (Generations)
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  
  // Decide grid step
  const gridStep = Math.max(1, Math.ceil(maxGen / 5));
  for (let g = 1; g <= maxGen; g += gridStep) {
    const x = getX(g);
    ctx.beginPath();
    ctx.moveTo(x, paddingTop);
    ctx.lineTo(x, paddingTop + chartHeight);
    ctx.stroke();
    
    ctx.fillText(g.toString(), x, paddingTop + chartHeight + 6);
  }
  
  // Draw labels and axes outline
  ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
  ctx.beginPath();
  ctx.moveTo(paddingLeft, paddingTop);
  ctx.lineTo(paddingLeft, paddingTop + chartHeight);
  ctx.lineTo(width - paddingRight, paddingTop + chartHeight);
  ctx.stroke();

  // 3. Plot Average Fitness (Blue line)
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

  // 4. Plot Best Fitness (Amber line with glowing shadow)
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

  // Legend
  ctx.font = '9px var(--font-sans)';
  ctx.textAlign = 'left';
  
  // Best Fitness Legend Dot
  ctx.fillStyle = '#f59e0b';
  ctx.beginPath();
  ctx.arc(paddingLeft + 10, paddingTop + 8, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.fillText('Best Fitness', paddingLeft + 18, paddingTop + 8);
  
  // Average Fitness Legend Dot
  ctx.fillStyle = '#6366f1';
  ctx.beginPath();
  ctx.arc(paddingLeft + 100, paddingTop + 8, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.fillText('Avg Fitness', paddingLeft + 108, paddingTop + 8);
}
