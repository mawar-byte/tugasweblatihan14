import { Bee } from './bee.js';

/**
 * Creates a new generation of bees using Selection, Crossover, and Mutation.
 * 
 * @param {Bee[]} currentBees - The completed generation of bees.
 * @param {number} popSize - Population size for the new generation.
 * @param {number} mutationRate - Probability of mutating a weight/bias [0, 1].
 * @param {number} mutationAmount - Speed/impact size of mutation changes.
 * @param {number} elitismPct - Percentage [0-30] of top bees to preserve unchanged.
 * @param {string} selectionMethod - 'tournament', 'roulette', or 'rank'.
 * @param {number} startX - Start X coordinate for the new bees.
 * @param {number} startY - Start Y coordinate for the new bees.
 * @param {number[]} brainTopology - Neural net layer size list.
 * @returns {Bee[]} - The new population of bees.
 */
export function nextGeneration(
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
  // 1. Sort bees by fitness descending
  const sortedBees = [...currentBees].sort((a, b) => b.fitness - a.fitness);
  const nextPop = [];

  // Calculate elitism count
  const eliteCount = Math.floor((popSize * elitismPct) / 100);
  
  // 2. Elitism: Keep the best performers unchanged
  for (let i = 0; i < eliteCount; i++) {
    if (sortedBees[i]) {
      // Re-create the bee at start, but keep its exact brain
      const eliteBee = new Bee(startX, startY, brainTopology, sortedBees[i].brain);
      nextPop.push(eliteBee);
    }
  }

  // Pre-calculate ranks if Rank Selection is used
  // Sorted from worst to best, so index 0 has rank 1, index N-1 has rank N
  const reversedBees = [...sortedBees].reverse();
  const totalRanks = (popSize * (popSize + 1)) / 2; // Sum from 1 to N

  // 3. Selection, Crossover, and Mutation for the rest of population
  while (nextPop.length < popSize) {
    const parentA = selectParent(sortedBees, reversedBees, totalRanks, selectionMethod);
    const parentB = selectParent(sortedBees, reversedBees, totalRanks, selectionMethod);
    
    // Crossover
    let childBrain = parentA.brain.crossover(parentB.brain);
    
    // Mutation
    childBrain.mutate(mutationRate, mutationAmount);
    
    // Create new child bee
    const childBee = new Bee(startX, startY, brainTopology, childBrain);
    nextPop.push(childBee);
  }

  return nextPop;
}

/**
 * Selects a parent bee from the population using the specified strategy.
 * 
 * @param {Bee[]} sortedBees - Bees sorted descending (index 0 is best).
 * @param {Bee[]} reversedBees - Bees sorted ascending (index 0 is worst).
 * @param {number} totalRanks - Sum of ranks for Rank Selection.
 * @param {string} method - Selection strategy.
 * @returns {Bee} - The selected parent bee.
 */
function selectParent(sortedBees, reversedBees, totalRanks, method) {
  const n = sortedBees.length;
  
  if (method === 'tournament') {
    // Tournament selection: Choose k random bees, pick the best one
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
    // Roulette Wheel Selection (Fitness proportionate)
    let totalFitness = sortedBees.reduce((sum, bee) => sum + bee.fitness, 0);
    
    if (totalFitness <= 0) {
      // Fallback: Random selection if all fitnesses are zero
      return sortedBees[Math.floor(Math.random() * n)];
    }
    
    const target = Math.random() * totalFitness;
    let currentSum = 0;
    
    for (let bee of sortedBees) {
      currentSum += bee.fitness;
      if (currentSum >= target) {
        return bee;
      }
    }
    
    return sortedBees[n - 1]; // Fallback
    
  } else if (method === 'rank') {
    // Rank Selection: Selection probability proportional to rank index
    const target = Math.random() * totalRanks;
    let currentSum = 0;
    
    for (let i = 0; i < n; i++) {
      const rank = i + 1; // Worst has rank 1, best has rank n
      currentSum += rank;
      if (currentSum >= target) {
        return reversedBees[i];
      }
    }
    
    return reversedBees[n - 1]; // Fallback
  }
  
  // Default fallback
  return sortedBees[Math.floor(Math.random() * n)];
}
