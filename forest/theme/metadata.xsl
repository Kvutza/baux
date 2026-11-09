<?xml version="1.0"?>
<!-- SPDX-License-Identifier: CC0-1.0 -->
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:fr="http://www.jonmsterling.com/jms-005P.xml">

  <xsl:template match="fr:month[.='1']">
    <xsl:text>January</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='2']">
    <xsl:text>February</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='3']">
    <xsl:text>March</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='4']">
    <xsl:text>April</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='5']">
    <xsl:text>May</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='6']">
    <xsl:text>June</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='7']">
    <xsl:text>July</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='8']">
    <xsl:text>August</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='9']">
    <xsl:text>September</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='10']">
    <xsl:text>October</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='11']">
    <xsl:text>November</xsl:text>
  </xsl:template>

  <xsl:template match="fr:month[.='12']">
    <xsl:text>December</xsl:text>
  </xsl:template>

  <xsl:template match="fr:year">
    <xsl:apply-templates />
  </xsl:template>

  <xsl:template match="fr:day">
    <xsl:apply-templates />
  </xsl:template>

  <xsl:template match="fr:date" mode="date-inner">
    <xsl:apply-templates select="fr:month" />
    <xsl:if test="fr:day">
      <xsl:text>&#160;</xsl:text>
      <xsl:apply-templates select="fr:day" />
    </xsl:if>
    <xsl:if test="fr:month">
      <xsl:text>,&#160;</xsl:text>
    </xsl:if>
    <xsl:apply-templates select="fr:year" />
  </xsl:template>

  <xsl:template match="fr:date[@href]">
    <li class="meta-item">
      <a class="link local" href="{@href}">
        <xsl:apply-templates select="." mode="date-inner" />
      </a>
    </li>
  </xsl:template>

  <xsl:template match="fr:date[not(@href)]">
    <li class="meta-item">
      <xsl:apply-templates select="." mode="date-inner" />
    </li>
  </xsl:template>

  <xsl:template match="fr:authors">
    <xsl:if test="fr:author or fr:contributor">
      <li class="meta-item">
        <address class="author">
          <xsl:for-each select="fr:author">
            <xsl:apply-templates />
            <xsl:if test="position()!=last()">
              <xsl:text>, &#x20;</xsl:text>
            </xsl:if>
          </xsl:for-each>
          <xsl:if test="fr:contributor">
            <xsl:text>&#x20;with contributions from&#x20;</xsl:text>
            <xsl:for-each select="fr:contributor">
              <xsl:apply-templates />
              <xsl:if test="position()!=last()">
                <xsl:text>,&#x20;</xsl:text>
              </xsl:if>
            </xsl:for-each>
          </xsl:if>
        </address>
      </li>
    </xsl:if>
  </xsl:template>

  <xsl:template match="fr:meta[@name='doi']">
    <li class="meta-item">
      <a class="doi link" href="{concat('https://www.doi.org/', .)}">
        <xsl:value-of select="." />
      </a>
    </li>
  </xsl:template>

  <xsl:template match="fr:meta[@name='orcid']">
    <li class="meta-item">
      <a class="orcid" href="{concat('https://orcid.org/', .)}">
        <xsl:value-of select="." />
      </a>
    </li>
  </xsl:template>

  <xsl:template match="fr:meta[@name='bibtex']">
    <pre>
      <xsl:value-of select="." />
    </pre>
  </xsl:template>

  <xsl:template match="fr:meta[@name='venue']|fr:meta[@name='position']|fr:meta[@name='institution']|fr:meta[@name='source']">
    <li class="meta-item">
      <xsl:apply-templates />
    </li>
  </xsl:template>

  <xsl:template match="fr:meta[@name='external']">
    <li class="meta-item">
      <a class="link external" href="{.}">
        <xsl:value-of select="." />
      </a>
    </li>
  </xsl:template>

  <xsl:template match="fr:meta[@name='slides']">
    <li class="meta-item">
      <a class="link external" href="{.}">
        <xsl:text>Slides</xsl:text>
      </a>
    </li>
  </xsl:template>

  <xsl:template match="fr:meta[@name='video']">
    <li class="meta-item">
      <a class="link external" href="{.}">
        <xsl:text>Video</xsl:text>
      </a>
    </li>
  </xsl:template>

</xsl:stylesheet>
