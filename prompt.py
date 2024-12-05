prompt = '''## Instructions
Rewrite the code and add a single line comment above `const [shouldRefreshGold, setShouldRefreshGold]`...

## Code

```ts
export default function Visualization() {
  const [instanceIdInputs, setInstanceIdInputs] = createSignal<
    InstanceId[] | null
  >(null);
  const [storedInput, setStoredInput] = createSignal<string>("");
  const [datapointOptions, setDatapointOptions] = createSignal<PropsInstance[]>(
    []
  );
  const [shouldRefreshGold, setShouldRefreshGold] =
    createSignal<boolean>(false);
  const [showGold, setShowGold] = createSignal<boolean>(false);
  const [selectedGoldRequestId, setSelectedGoldRequestId] = createSignal<
    string | undefined
  >(undefined);
  const [goldInstances, setGoldInstances] = createSignal<
    {
      sessionId: string;
      email: string | undefined;
      requestId: string | undefined;
      dateAdded: Date;
      type: $Enums.CppGoldExampleType;
    }[]
  >([]);
}
```

## Rewritten code

```ts'''