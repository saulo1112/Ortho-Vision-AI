import { StyleSheet, Text, View } from 'react-native';

import type { Detection } from '../api/types';
import { percent } from '../lib/format';
import { classStyle } from '../theme/classes';
import { colors, fonts, radius } from '../theme/tokens';

/** One detected implant instance: class, confidence bar, percentage. */
export function DetectionCard({ detection }: { detection: Detection }) {
  const { label, color } = classStyle(detection.class_name);
  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View style={[styles.dot, { backgroundColor: color }]} />
        <View style={styles.titleBlock}>
          <Text style={styles.title}>{label}</Text>
          <Text style={styles.subtitle}>{detection.class_name}</Text>
        </View>
        <Text style={[styles.percent, { color }]}>{percent(detection.confidence)}</Text>
      </View>
      <View style={styles.track}>
        <View
          style={[
            styles.fill,
            { width: `${Math.round(detection.confidence * 100)}%`, backgroundColor: color },
          ]}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: 16,
    borderRadius: radius.control,
    backgroundColor: colors.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.surfaceBorder,
    gap: 12,
  },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  titleBlock: { flex: 1 },
  title: { color: colors.text, fontSize: 15, fontWeight: '600', fontFamily: fonts?.sans },
  subtitle: { color: colors.textTertiary, fontSize: 12, fontFamily: fonts?.mono },
  percent: { fontSize: 17, fontWeight: '700', fontFamily: fonts?.rounded },
  track: {
    height: 5,
    borderRadius: 3,
    backgroundColor: 'rgba(148, 197, 233, 0.12)',
    overflow: 'hidden',
  },
  fill: { height: '100%', borderRadius: 3 },
});
