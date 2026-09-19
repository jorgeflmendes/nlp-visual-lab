export interface TraceTensor {
  name: string;
  shape: number[];
  values: number[] | Float32Array;
  dtype?: string;
  axes?: string[];
  description?: string;
}

export interface TraceToken {
  text: string;
  id?: number;
  position?: number;
}

export interface Candidate {
  token: string;
  id?: number;
  probability: number;
  logit?: number;
  selected?: boolean;
}

export interface ExecutionStep {
  name: string;
  stage: string;
  operation: string;
  description: string;
  formula?: string;
  tensors: TraceTensor[];
  tokens?: TraceToken[];
  selectedToken?: number;
  probabilities?: Candidate[];
  attention?: { source: string[]; target: string[]; weights: number[][]; row: number };
}

export interface ExecutionTrace {
  output: string;
  outputLabel: string;
  backend: string;
  elapsed: number;
  steps: ExecutionStep[];
  note?: string;
}

export interface StepGroup {
  stage: string;
  startIndex: number;
  endIndex: number;
  steps: { step: ExecutionStep; index: number }[];
}

export function groupExecutionSteps(steps: ExecutionStep[]): StepGroup[] {
  const groups: StepGroup[] = [];
  for (let index = 0; index < steps.length; index++) {
    const step = steps[index];
    const lastGroup = groups[groups.length - 1];
    if (lastGroup && lastGroup.stage === step.stage) {
      lastGroup.endIndex = index;
      lastGroup.steps.push({ step, index });
    } else {
      groups.push({
        stage: step.stage,
        startIndex: index,
        endIndex: index,
        steps: [{ step, index }],
      });
    }
  }
  return groups;
}

