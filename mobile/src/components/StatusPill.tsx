import { useEffect, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { getHealth } from '../api/client';
import { colors, fonts, radius } from '../theme/tokens';

type State = 'checking' | 'online' | 'offline';

/** Small server-status indicator; free-tier hosts sleep, so surface that. */
export function StatusPill() {
  const [state, setState] = useState<State>('checking');

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        await getHealth();
        if (!cancelled) setState('online');
      } catch {
        if (!cancelled) setState('offline');
      }
    };
    check();
    const interval = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const dotColor =
    state === 'online' ? colors.online : state === 'offline' ? colors.danger : colors.textTertiary;
  const label =
    state === 'online' ? 'Model online' : state === 'offline' ? 'Server waking up…' : 'Connecting…';

  return (
    <View style={styles.pill}>
      <View style={[styles.dot, { backgroundColor: dotColor }]} />
      <Text style={styles.label}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: radius.pill,
    backgroundColor: colors.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.surfaceBorder,
    alignSelf: 'flex-start',
  },
  dot: { width: 7, height: 7, borderRadius: 4 },
  label: { color: colors.textSecondary, fontSize: 12, fontFamily: fonts?.sans },
});
